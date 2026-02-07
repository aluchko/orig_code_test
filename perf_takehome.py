#!/usr/bin/env python3

"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(
        self,
        slots: list[tuple[Engine, tuple] | dict[str, list[tuple]]],
        vliw: bool = False,
    ):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        if not vliw:
            for slot_entry in slots:
                if isinstance(slot_entry, dict):
                    instrs.append(slot_entry)
                    continue
                engine, slot = slot_entry
                instrs.append({engine: [slot]})
            return instrs

        def addr_range(addr: int, length: int = 1) -> set[int]:
            return set(range(addr, addr + length))

        def slot_reads_writes(engine: str, slot: tuple) -> tuple[set[int], set[int]]:
            op = slot[0]
            if engine == "debug":
                if op == "compare":
                    return {slot[1]}, set()
                if op == "vcompare":
                    return addr_range(slot[1], VLEN), set()
                return set(), set()
            if engine == "flow":
                if op == "pause":
                    return set(), set()
                # vselect dest, cond, a, b
                if op == "vselect":
                    return (
                        addr_range(slot[2], VLEN)
                        | addr_range(slot[3], VLEN)
                        | addr_range(slot[4], VLEN),
                        addr_range(slot[1], VLEN),
                    )
                return set(slot[2:]), {slot[1]}
            if engine == "store":
                # store/vstore addr, val
                if op == "vstore":
                    return {slot[1]} | addr_range(slot[2], VLEN), set()
                return set(slot[1:]), set()
            if engine == "load":
                # const dest, imm | load/vload dest, addr | load_offset dest, base, offset
                if op == "const":
                    return set(), {slot[1]}
                if op == "load_offset":
                    offset = slot[3]
                    return {slot[2] + offset}, {slot[1] + offset}
                if op == "vload":
                    return {slot[2]}, addr_range(slot[1], VLEN)
                return {slot[2]}, {slot[1]}
            # alu/valu: op dest, src1, src2
            if engine == "valu":
                if op == "vbroadcast":
                    return {slot[2]}, addr_range(slot[1], VLEN)
                if op == "multiply_add":
                    return (
                        addr_range(slot[2], VLEN)
                        | addr_range(slot[3], VLEN)
                        | addr_range(slot[4], VLEN),
                        addr_range(slot[1], VLEN),
                    )
                return (
                    addr_range(slot[2], VLEN) | addr_range(slot[3], VLEN),
                    addr_range(slot[1], VLEN),
                )
            return set(slot[2:]), {slot[1]}

        def can_pack(
            bundle_reads: set[int],
            bundle_writes: set[int],
            reads: set[int],
            writes: set[int],
        ) -> bool:
            if reads & bundle_writes:
                return False
            if writes & (bundle_reads | bundle_writes):
                return False
            return True

        current = {}
        bundle_reads: set[int] = set()
        bundle_writes: set[int] = set()

        def flush():
            nonlocal current, bundle_reads, bundle_writes
            if current:
                instrs.append(current)
                current = {}
                bundle_reads = set()
                bundle_writes = set()

        for slot_entry in slots:
            if isinstance(slot_entry, dict):
                flush()
                instrs.append(slot_entry)
                continue

            engine, slot = slot_entry
            reads, writes = slot_reads_writes(engine, slot)
            engine_slots = current.get(engine, [])
            if (
                len(engine_slots) >= SLOT_LIMITS[engine]
                or not can_pack(bundle_reads, bundle_writes, reads, writes)
            ):
                flush()
                engine_slots = current.get(engine, [])
            engine_slots.append(slot)
            current[engine] = engine_slots
            bundle_reads |= reads
            bundle_writes |= writes

        flush()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        pipe_buffers = 3
        vec_idx = [
            self.alloc_scratch(f"vec_idx_{i}", length=VLEN) for i in range(pipe_buffers)
        ]
        vec_val = [
            self.alloc_scratch(f"vec_val_{i}", length=VLEN) for i in range(pipe_buffers)
        ]
        vec_node_val = [
            self.alloc_scratch(f"vec_node_val_{i}", length=VLEN)
            for i in range(pipe_buffers)
        ]
        vec_addr = [
            self.alloc_scratch(f"vec_addr_{i}", length=VLEN) for i in range(pipe_buffers)
        ]
        vec_tmp1 = [
            self.alloc_scratch(f"vec_tmp1_{i}", length=VLEN)
            for i in range(pipe_buffers)
        ]
        vec_tmp2 = [
            self.alloc_scratch(f"vec_tmp2_{i}", length=VLEN)
            for i in range(pipe_buffers)
        ]
        vec_tmp3 = [
            self.alloc_scratch(f"vec_tmp3_{i}", length=VLEN)
            for i in range(pipe_buffers)
        ]
        tmp_addr_idx = [
            self.alloc_scratch(f"tmp_addr_idx_{i}") for i in range(pipe_buffers)
        ]
        tmp_addr_val = [
            self.alloc_scratch(f"tmp_addr_val_{i}") for i in range(pipe_buffers)
        ]
        vec_const_map: dict[int, int] = {}

        def get_vec_const(val: int, name: str | None = None) -> int:
            if val not in vec_const_map:
                addr = self.alloc_scratch(name, length=VLEN)
                scalar_addr = self.scratch_const(val)
                self.add("valu", ("vbroadcast", addr, scalar_addr))
                vec_const_map[val] = addr
            return vec_const_map[val]

        vec_zero = get_vec_const(0, name="vec_zero")
        vec_one = get_vec_const(1, name="vec_one")
        vec_forest_base = self.alloc_scratch("vec_forest_base", length=VLEN)
        vec_n_nodes = self.alloc_scratch("vec_n_nodes", length=VLEN)
        self.add("valu", ("vbroadcast", vec_forest_base, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", vec_n_nodes, self.scratch["n_nodes"]))

        def build_load_stage(round: int, batch_base: int, buf: int) -> list[tuple]:
            i_const = self.scratch_const(batch_base)
            return [
                ("alu", ("+", tmp_addr_idx[buf], self.scratch["inp_indices_p"], i_const)),
                ("alu", ("+", tmp_addr_val[buf], self.scratch["inp_values_p"], i_const)),
                ("load", ("vload", vec_idx[buf], tmp_addr_idx[buf])),
                ("load", ("vload", vec_val[buf], tmp_addr_val[buf])),
            ]

        def build_compute_stage(round: int, batch_base: int, buf: int) -> list[tuple]:
            slots = [
                (
                    "debug",
                    (
                        "vcompare",
                        vec_idx[buf],
                        [(round, batch_base + vi, "idx") for vi in range(VLEN)],
                    ),
                ),
                (
                    "debug",
                    (
                        "vcompare",
                        vec_val[buf],
                        [(round, batch_base + vi, "val") for vi in range(VLEN)],
                    ),
                ),
                ("valu", ("+", vec_addr[buf], vec_forest_base, vec_idx[buf])),
            ]
            for vi in range(0, VLEN, 2):
                slots.append(("load", ("load_offset", vec_node_val[buf], vec_addr[buf], vi)))
                slots.append(
                    ("load", ("load_offset", vec_node_val[buf], vec_addr[buf], vi + 1))
                )
            slots.append(
                (
                    "debug",
                    (
                        "vcompare",
                        vec_node_val[buf],
                        [(round, batch_base + vi, "node_val") for vi in range(VLEN)],
                    ),
                )
            )
            slots.append(("valu", ("^", vec_val[buf], vec_val[buf], vec_node_val[buf])))
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                slots.append(
                    ("valu", (op1, vec_tmp1[buf], vec_val[buf], get_vec_const(val1)))
                )
                slots.append(
                    ("valu", (op3, vec_tmp2[buf], vec_val[buf], get_vec_const(val3)))
                )
                slots.append(("valu", (op2, vec_val[buf], vec_tmp1[buf], vec_tmp2[buf])))
            slots.append(("valu", ("&", vec_tmp1[buf], vec_val[buf], vec_one)))
            slots.append(("valu", ("+", vec_tmp3[buf], vec_tmp1[buf], vec_one)))
            slots.append(("valu", ("<<", vec_idx[buf], vec_idx[buf], vec_one)))
            slots.append(("valu", ("+", vec_idx[buf], vec_idx[buf], vec_tmp3[buf])))
            slots.append(("valu", ("<", vec_tmp1[buf], vec_idx[buf], vec_n_nodes)))
            slots.append(("valu", ("*", vec_idx[buf], vec_idx[buf], vec_tmp1[buf])))
            return slots

        def build_store_stage(round: int, batch_base: int, buf: int) -> list[tuple]:
            return [
                ("store", ("vstore", tmp_addr_idx[buf], vec_idx[buf])),
                ("store", ("vstore", tmp_addr_val[buf], vec_val[buf])),
            ]

        for round in range(rounds):
            vector_end = batch_size - (batch_size % VLEN)
            # We don't have to worry about the tail since batch_size is always a multiple of VLEN.
            batch_bases = list(range(0, vector_end, VLEN))
            load_stages = []
            compute_stages = []
            store_stages = []
            for batch_base in batch_bases:
                buf = (batch_base // VLEN) % pipe_buffers
                load_stages.append(build_load_stage(round, batch_base, buf))
                compute_stages.append(build_compute_stage(round, batch_base, buf))
                store_stages.append(build_store_stage(round, batch_base, buf))

            for i in range(len(batch_bases) + 2):
                if i < len(batch_bases):
                    body.extend(load_stages[i])
                if 0 <= i - 1 < len(batch_bases):
                    body.extend(compute_stages[i - 1])
                if 0 <= i - 2 < len(batch_bases):
                    body.extend(store_stages[i - 2])

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
