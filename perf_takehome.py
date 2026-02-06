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

    def build(self, slots: list[tuple[Engine, tuple] | dict[str, list[tuple]]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for slot_entry in slots:
            if isinstance(slot_entry, dict):
                instrs.append(slot_entry)
                continue
            engine, slot = slot_entry
            instrs.append({engine: [slot]})
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

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr_idx = self.alloc_scratch("tmp_addr_idx")
        tmp_addr_val = self.alloc_scratch("tmp_addr_val")

        vec_idx = self.alloc_scratch("vec_idx", length=VLEN)
        vec_val = self.alloc_scratch("vec_val", length=VLEN)
        vec_node_val = self.alloc_scratch("vec_node_val", length=VLEN)
        vec_addr = self.alloc_scratch("vec_addr", length=VLEN)
        vec_tmp1 = self.alloc_scratch("vec_tmp1", length=VLEN)
        vec_tmp2 = self.alloc_scratch("vec_tmp2", length=VLEN)
        vec_tmp3 = self.alloc_scratch("vec_tmp3", length=VLEN)
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
        vec_two = get_vec_const(2, name="vec_two")
        vec_forest_base = self.alloc_scratch("vec_forest_base", length=VLEN)
        vec_n_nodes = self.alloc_scratch("vec_n_nodes", length=VLEN)
        self.add("valu", ("vbroadcast", vec_forest_base, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", vec_n_nodes, self.scratch["n_nodes"]))

        for round in range(rounds):
            vector_end = batch_size - (batch_size % VLEN)
            # We don't have to worry about the tail since batch_size is always a multiple of VLEN.
            for batch_base in range(0, vector_end, VLEN):
                i_const = self.scratch_const(batch_base)
                # idx = mem[inp_indices_p + i : i + VLEN]
                body.append(
                    {
                        "alu": [
                            ("+", tmp_addr_idx, self.scratch["inp_indices_p"], i_const),
                            ("+", tmp_addr_val, self.scratch["inp_values_p"], i_const),
                        ]
                    }
                )
                body.append(
                    {
                        "load": [
                            ("vload", vec_idx, tmp_addr_idx),
                            ("vload", vec_val, tmp_addr_val),
                        ]
                    }
                )
                body.append(
                    {
                        "debug": [
                            (
                                "vcompare",
                                vec_idx,
                                [(round, batch_base + vi, "idx") for vi in range(VLEN)],
                            ),
                            (
                                "vcompare",
                                vec_val,
                                [(round, batch_base + vi, "val") for vi in range(VLEN)],
                            ),
                        ]
                    }
                )
                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", vec_addr, vec_forest_base, vec_idx))) # vec_addr = vec_forest_base + vec_idx
                for vi in range(0, VLEN, 2):
                    body.append(
                        {
                            "load": [
                                ("load_offset", vec_node_val, vec_addr, vi),
                                ("load_offset", vec_node_val, vec_addr, vi + 1),
                            ]
                        }
                    )
                body.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            vec_node_val,
                            [(round, batch_base + vi, "node_val") for vi in range(VLEN)],
                        ),
                    )
                )
                # val = myhash(val ^ node_val), this is so we know which branch of the tree to take.
                body.append(("valu", ("^", vec_val, vec_val, vec_node_val)))
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    body.append(
                        {
                            "valu": [
                                (op1, vec_tmp1, vec_val, get_vec_const(val1)),
                                (op3, vec_tmp2, vec_val, get_vec_const(val3)),
                            ]
                        }
                    )
                    body.append(("valu", (op2, vec_val, vec_tmp1, vec_tmp2)))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", vec_tmp1, vec_val, vec_two)))
                body.append(("valu", ("==", vec_tmp1, vec_tmp1, vec_zero)))
                body.append(
                    {
                        "flow": [("vselect", vec_tmp3, vec_tmp1, vec_one, vec_two)],
                        "valu": [("*", vec_idx, vec_idx, vec_two)],
                    }
                )
                body.append(("valu", ("+", vec_idx, vec_idx, vec_tmp3)))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", vec_tmp1, vec_idx, vec_n_nodes)))
                body.append(("flow", ("vselect", vec_idx, vec_tmp1, vec_idx, vec_zero)))
                # mem[inp_indices_p + i : i + VLEN] = idx
                body.append(
                    {
                        "alu": [
                            ("+", tmp_addr_idx, self.scratch["inp_indices_p"], i_const),
                            ("+", tmp_addr_val, self.scratch["inp_values_p"], i_const),
                        ]
                    }
                )
                body.append(
                    {
                        "store": [
                            ("vstore", tmp_addr_idx, vec_idx),
                            ("vstore", tmp_addr_val, vec_val),
                        ]
                    }
                )
                # mem[inp_values_p + i : i + VLEN] = val

        body_instrs = self.build(body)
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
