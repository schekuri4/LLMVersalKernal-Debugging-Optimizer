# Data Sources & Citations

This document lists every data source used to build the Versal Kernel Debugging
training dataset (`super_hls_train.jsonl` / `super_hls_eval.jsonl`).

---

## 1. HLStrans — HLS Code Optimization Pairs

| Field         | Value                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------- |
| **Records**   | 124,208                                                                                        |
| **Format**    | Concatenated JSON (`instruction` / `input` / `output`)                                         |
| **Content**   | Source-to-source C → optimized HLS C with pragmas                                              |
| **File**      | `datasets/HLStrans.jsonl`                                                                      |
| **Reference** | Collini, S., Sterin, B., et al. _"HLStrans: A Large-Scale Dataset for HLS Code Optimization."_ |

---

## 2. ForgeHLS — QoR Prediction Data

| Field         | Value                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------ |
| **Records**   | 291,966 (capped to 80,000 in dataset)                                                            |
| **Format**    | JSON array (`instruction` / `input` / `output`)                                                  |
| **Content**   | HLS code → latency/resource prediction                                                           |
| **Repo**      | https://github.com/zedong-peng/ForgeHLS                                                          |
| **File**      | `data/forge_hls_exported.json` (merged from gzipped splits)                                      |
| **Reference** | Peng, Z., et al. _"ForgeHLS: LLM-Driven HLS Code Generation and Quality of Results Prediction."_ |

---

## 3. CHStone Benchmarks

| Field         | Value                                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Files**     | 63 C/C++ source files                                                                                                                                   |
| **Content**   | Classic HLS benchmark suite (ADPCM, AES, Blowfish, DFADD, DFDIV, DFMUL, DFSIN, GSM, JPEG, MIPS, Motion, SHA)                                            |
| **Repo**      | https://github.com/ferrandi/CHStone                                                                                                                     |
| **Reference** | Hara, Y., Tomiyama, H., Honda, S., Takada, H., Ishii, K. _"CHStone: A Benchmark Program Suite for Practical C-Based High-Level Synthesis."_ ISCAS 2008. |

---

## 4. MachSuite Benchmarks

| Field         | Value                                                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Files**     | 90 C/C++ source files                                                                                                                               |
| **Content**   | HLS-oriented benchmark kernels (AES, Backprop, BFS, FFT, Gemm, KMP, MD, NW, Sort, Spmv, Stencil, Viterbi)                                           |
| **Repo**      | https://github.com/breagen/MachSuite                                                                                                                |
| **Reference** | Reagen, B., Harvard, R., Weisz, G.Y., Achour, S., et al. _"MachSuite: Benchmarks for Accelerator Design and Customized Architectures."_ IISWC 2014. |

---

## 5. Vitis HLS Introductory Examples

| Field       | Value                                                                                                      |
| ----------- | ---------------------------------------------------------------------------------------------------------- |
| **Files**   | 290 C/C++ source files                                                                                     |
| **Content** | Official Xilinx introductory examples for Vitis HLS (interface, loop optimization, dataflow, pragma usage) |
| **Repo**    | https://github.com/Xilinx/Vitis-HLS-Introductory-Examples                                                  |
| **License** | Apache 2.0                                                                                                 |

---

## 6. HLSyn — Graph Neural Network HLS Benchmarks

| Field         | Value                                                                                 |
| ------------- | ------------------------------------------------------------------------------------- |
| **Files**     | 42 kernel files                                                                       |
| **Content**   | HLS kernel source files annotated with synthesis results for GNN-based QoR prediction |
| **Repo**      | https://github.com/ZongyueQin/HLSyn                                                   |
| **Reference** | Qin, Z., et al. _"HLSyn: High-Level Synthesis Dataset for Machine Learning."_         |

---

## 7. Vitis Acceleration Examples

| Field         | Value                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------- |
| **Files**     | 110 C/C++ source files                                                                       |
| **Git pairs** | 662 before/after code pairs mined from commit history                                        |
| **Content**   | Production Vitis acceleration examples (host + kernel code), real bug fixes and improvements |
| **Repo**      | https://github.com/Xilinx/Vitis_Accel_Examples                                               |
| **License**   | Apache 2.0                                                                                   |

---

## 8. Vitis Libraries

| Field         | Value                                                                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Files**     | 5,203 C/C++ source files                                                                                                                          |
| **Git pairs** | 942 before/after code pairs mined from commit history                                                                                             |
| **Content**   | Xilinx open-source library for BLAS, codec, data analytics, database, DSP, graph, quantitative finance, security, solver, sparse, utility, vision |
| **Repo**      | https://github.com/Xilinx/Vitis_Libraries                                                                                                         |
| **License**   | Apache 2.0                                                                                                                                        |

---

## 9. Vitis Tutorials

| Field         | Value                                                                       |
| ------------- | --------------------------------------------------------------------------- |
| **Files**     | 1,451 C/C++ source files                                                    |
| **Git pairs** | 292 before/after code pairs mined from commit history                       |
| **Content**   | Official Xilinx tutorials for Vitis, AI Engine, HLS, and hardware emulation |
| **Repo**      | https://github.com/Xilinx/Vitis-Tutorials                                   |
| **License**   | Apache 2.0                                                                  |

---

## 10. HeteroCL

| Field         | Value                                                                                                                                  |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Files**     | 17 C/C++ source files                                                                                                                  |
| **Git pairs** | 248 before/after code pairs mined from commit history                                                                                  |
| **Content**   | Heterogeneous computing DSL with HLS backend; real development bug fixes and refactors                                                 |
| **Repo**      | https://github.com/cornell-zhang/heterocl                                                                                              |
| **Reference** | Lai, Y.-H., et al. _"HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing."_ FPGA 2019. |

---

## 11. Merlin Compiler (UCLA-VAST)

| Field         | Value                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------ |
| **Git pairs** | 47 before/after code pairs mined from commit history                                             |
| **Content**   | Source-to-source HLS optimization compiler; real compiler development fixes                      |
| **Repo**      | https://github.com/UCLA-VAST/Merlin-UCLA                                                         |
| **Reference** | Cong, J., Fang, Z., Gill, M., Reinman, G. _"Merlin: An Open-Source FPGA Programming Framework."_ |

---

## 12. SODA-OPT

| Field         | Value                                                 |
| ------------- | ----------------------------------------------------- |
| **Git pairs** | 359 before/after code pairs mined from commit history |
| **Content**   | MLIR-based HLS optimization and synthesis framework   |
| **Repo**      | https://github.com/pnnl/soda-opt                      |

---

## 13. Xilinx Runtime (XRT)

| Field         | Value |
| ------------- | ----- |
| **Files**     | 1,496 C/C++ source files |
| **Git pairs** | 1,138 before/after code pairs mined from commit history |
| **Content**   | Xilinx Runtime library — host-side APIs, kernel management, device drivers for FPGA acceleration |
| **Repo**      | https://github.com/Xilinx/XRT |
| **License**   | Apache 2.0 |

---

## 14. AutoBridge (UCLA-VAST)

| Field         | Value |
| ------------- | ----- |
| **Files**     | 372 C/C++ source files |
| **Git pairs** | 1 before/after code pair |
| **Content**   | Automated floorplanning for multi-die FPGA HLS designs |
| **Repo**      | https://github.com/UCLA-VAST/AutoBridge |
| **Reference** | Guo, L., et al. _"AutoBridge: Coupling Coarse-Grained Floorplanning and Pipelining for High-Frequency HLS Design on Multi-Die FPGAs."_ FPGA 2021. |

---

## 15. Vitis AI

| Field         | Value |
| ------------- | ----- |
| **Files**     | 983 C/C++ source files |
| **Git pairs** | 52 before/after code pairs mined from commit history |
| **Content**   | Xilinx AI inference engine for DPU/AI Engine on Versal and Zynq platforms |
| **Repo**      | https://github.com/Xilinx/Vitis-AI |
| **License**   | Apache 2.0 |

---

## 16. Odyssey (UCLA-VAST)

| Field       | Value |
| ----------- | ----- |
| **Files**   | 17 C/C++ source files |
| **Content** | HLS kernel optimization research tool |
| **Repo**    | https://github.com/UCLA-VAST/Odyssey |

---

## 17. OpenFPGA

| Field         | Value |
| ------------- | ----- |
| **Files**     | 692 C/C++ source files |
| **Git pairs** | 941 before/after code pairs mined from commit history |
| **Content**   | Open-source FPGA architecture and CAD framework |
| **Repo**      | https://github.com/LNIS-Projects/OpenFPGA |
| **Reference** | Tang, X., et al. _"OpenFPGA: An Open-Source Framework for Agile Prototyping Customizable FPGAs."_ |

---

## 18. MLIR-AIE (Xilinx)

| Field         | Value |
| ------------- | ----- |
| **Files**     | 680 C/C++ source files |
| **Git pairs** | 201 before/after code pairs mined from commit history |
| **Content**   | MLIR-based compiler for Xilinx AI Engine — generates AIE code for Versal ACAP |
| **Repo**      | https://github.com/Xilinx/mlir-aie |
| **License**   | Apache 2.0 |

---

## 19. VTR — Verilog-to-Routing

| Field         | Value |
| ------------- | ----- |
| **Files**     | 2,978 C/C++ source files |
| **Git pairs** | 1,091 before/after code pairs mined from commit history |
| **Content**   | Academic FPGA CAD toolchain (synthesis, packing, placement, routing) |
| **Repo**      | https://github.com/verilog-to-routing/vtr-verilog-to-routing |
| **Reference** | Murray, K.E., et al. _"VTR 8: High-Performance CAD and Customizable FPGA Architecture Modelling."_ ACM TRETS, 2020. |

---

## 20. Known Real-World HLS Bug Patterns (Hand-Curated)

| Field          | Value |
| -------------- | ----- |
| **Patterns**   | 50 unique bug/fix pairs |
| **Samples**    | 550 (11× instruction-varied) |
| **Content**    | Hand-written (buggy code, fixed code, explanation) triples based on documented Xilinx AR notes, Vitis HLS known issues, and common HLS anti-patterns |
| **Categories** | Pipeline conflicts, AXI bundle contention, dataflow violations, stream deadlocks, ap_int overflow, unsynthesizable constructs (malloc, std::vector), Vivado→Vitis migration, AIE/Versal-specific issues, burst inference, BRAM port mismatch, convolution line buffers, histogram dependencies, FIR filters, dot product FP latency hiding, struct packing |

---

## 21. Synthetic Debugging Samples (AI-Generated Corruptions)

| Field        | Value                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Samples**  | ~370,000                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Content**  | Programmatic corruption of gold HLS code using 21 distinct modes                                                                                                                                                                                                                                                                                                                                                 |
| **Modes**    | `strip_pragmas`, `wrong_pipeline_ii`, `remove_unroll`, `break_interface`, `wrong_data_type`, `off_by_one`, `remove_pipeline`, `swap_read_write`, `remove_loop_bound`, `add_dead_code`, `shuffle_statements`, `break_dependency`, `wrong_operator`, `misplace_pragma`, `remove_dataflow`, `break_array_partition`, `wrong_bind_storage`, `remove_array_reshape`, `bad_fifo_depth`, `remove_inline`, `partial_fix` |
| **Variants** | Multi-bug, single-bug, explain-the-bug (with structured reasoning output)                                                                                                                                                                                                                                                                                                                                        |

---

## Dataset Summary

| Category                       | Records (1 cycle)    | Records (3 cycles) | Source Type       |
| ------------------------------ | -------------------- | ------------------- | ----------------- |
| HLS Optimization (HLStrans)    | 124,208              | 124,208             | Academic dataset  |
| QoR Prediction (ForgeHLS)      | 80,000               | 80,000              | Academic dataset  |
| C/C++ Benchmarks               | 7,760                | 7,760               | Open-source repos |
| Debug — multi-bug corruptions  | ~323,000             | ~969,000            | Synthetic         |
| Debug — single-bug corruptions | ~31,000              | ~93,000             | Synthetic         |
| Debug — explain-the-bug        | ~18,600              | ~56,000             | Synthetic         |
| Debug — real git fixes         | 6,768                | 6,768               | Mined from 18 repos |
| Debug — known issue patterns   | 550                  | 550                 | Hand-curated (50 patterns) |
| **Total (1 cycle)**            | **~592,000**         |                     |                   |
| **Total (3 cycles, merged)**   |                      | **~1,380,000**      |                   |

Git-mined fix breakdown by repo:
| Repo | Pairs |
| ---- | ----- |
| XRT | 1,138 |
| vtr-verilog-to-routing | 1,091 |
| Vitis_Libraries | 942 |
| OpenFPGA | 941 |
| Vitis_Accel_Examples | 662 |
| soda-opt | 359 |
| Vitis-Tutorials | 292 |
| heterocl | 248 |
| mlir-aie | 201 |
| Vitis-AI | 52 |
| Merlin-UCLA | 47 |
| **Total** | **6,768** |

---

## Build Reproducibility

```bash
# Single cycle (default):
python build_super_dataset.py

# Multi-cycle (3 runs with different seeds, merged + deduplicated):
python build_super_dataset.py --cycles 3
```

Requires:

- `datasets/HLStrans.jsonl`
- `data/forge_hls_exported.json`
- `data/raw_benchmarks/` (populated from cloned repos)
- `repos/` directory with cloned git repositories (for mining)

Output: `super_hls_train.jsonl` + `super_hls_eval.jsonl`
