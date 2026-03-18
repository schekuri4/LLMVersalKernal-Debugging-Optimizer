"""
Versal Kernel Debugging Dataset Builder
=======================================
Primary goal: train an LLM to **debug** Versal / Vitis HLS kernels.

Data mix (target proportions):
  ~45 % Debugging  – corrupted HLS code → fixed code
  ~30 % Optimization – slow C → optimized HLS
  ~20 % QoR Understanding – code → latency/resource prediction (capped)
   ~5 % Eval hold-out (written to a separate file)

Inputs:
  - datasets/HLStrans.jsonl  (124k S2S optimization pairs)
  - data/forge_hls_exported.json (292k QoR prediction records)
  - data/raw_benchmarks/  (.c/.cpp from CHStone/MachSuite/VitisHLS/HLSyn)

Outputs:
  - super_hls_train.jsonl  (training set)
  - super_hls_eval.jsonl   (held-out evaluation set)
"""

import json
import os
import re
import random
import glob
import hashlib
import subprocess
import sys

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILES = {
    "hlstrans":   os.path.join(BASE_DIR, "datasets", "HLStrans.jsonl"),
    "forgehls":   os.path.join(BASE_DIR, "data", "forge_hls_exported.json"),
    "benchmarks": os.path.join(BASE_DIR, "data", "raw_benchmarks"),
}

OUTPUT_TRAIN = os.path.join(BASE_DIR, "super_hls_train.jsonl")
OUTPUT_EVAL  = os.path.join(BASE_DIR, "super_hls_eval.jsonl")

# ── Tuning knobs ──────────────────────────────────────────
NEGATIVE_RATIO  = 0.65         # 65 % of eligible gold → debug source pool
NEGATIVE_PASSES = 4            # 4 corruption variants per sampled entry
FORGE_QOR_CAP   = 80_000       # Cap QoR records so they don't dominate
EVAL_FRACTION   = 0.05         # 5 % held out for evaluation
RANDOM_SEED     = 42

random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────
# CORRUPTER  –  turns good HLS into broken "software" code
# ──────────────────────────────────────────────────────────
# Distinct corruption strategies — each produces a different class of bug
CORRUPTION_MODES = [
    # --- Generic HLS bugs ---
    "strip_pragmas",
    "revert_types",
    "inject_malloc",
    "inject_recursion",
    "swap_stream",
    "remove_pipeline",
    "add_pointer_cast",
    "variable_length_array",
    "global_variable",
    "missing_interface",
    "wrong_loop_bound",
    "double_free",
    "new_delete",
    "float_compare",
    # --- Versal / Vitis HLS specific bugs ---
    "remove_dataflow",
    "break_array_partition",
    "wrong_bind_storage",
    "remove_array_reshape",
    "bad_fifo_depth",
    "remove_inline",
    # --- Partial-fix (some pragmas correct, others wrong) ---
    "partial_fix",
]

def corrupt_hls_code(code: str, mode: str = None) -> str:
    """
    Creates 'Negative' data by degrading hardware-aware C/C++ back into
    naive software C that will fail or bottleneck during HLS synthesis.
    If *mode* is None a random subset of corruptions is applied.
    """
    if mode is None:
        # Apply 2-4 random corruptions
        modes = random.sample(CORRUPTION_MODES,
                              k=random.randint(2, min(4, len(CORRUPTION_MODES))))
    else:
        modes = [mode]

    corrupted = code

    for m in modes:
        if m == "strip_pragmas":
            corrupted = re.sub(r'#pragma\s+HLS\s+[^\n]*', '', corrupted)

        elif m == "revert_types":
            corrupted = corrupted.replace('ap_int', 'int')
            corrupted = corrupted.replace('ap_uint', 'unsigned int')
            corrupted = re.sub(r'ap_fixed<[^>]*>', 'float', corrupted)

        elif m == "inject_malloc":
            corrupted = ("// BUG: dynamic allocation is not synthesizable\n"
                         + corrupted)
            corrupted = corrupted.replace(
                '{', '{\n    int* tmp = (int*)malloc(64);', 1)

        elif m == "inject_recursion":
            # Wrap body in a fake recursive helper
            corrupted = ("// BUG: recursion is not synthesizable\n"
                         "void _recursive_helper(int n) {\n"
                         "    if (n <= 0) return;\n"
                         "    _recursive_helper(n - 1);\n"
                         "}\n\n" + corrupted)

        elif m == "swap_stream":
            corrupted = corrupted.replace('hls::stream', 'std::queue')

        elif m == "remove_pipeline":
            corrupted = re.sub(
                r'#pragma\s+HLS\s+PIPELINE[^\n]*', '', corrupted)
            corrupted = re.sub(
                r'#pragma\s+HLS\s+pipeline[^\n]*', '', corrupted)

        elif m == "add_pointer_cast":
            corrupted = corrupted.replace(
                '{', '{\n    void* _bad = (void*)0xDEAD; // non-synthesizable ptr\n', 1)

        elif m == "variable_length_array":
            corrupted = ("// BUG: VLA not supported in HLS\n"
                         + corrupted)
            corrupted = corrupted.replace(
                '{', '{\n    int vla_size = 128;\n    int vla_arr[vla_size];\n', 1)

        elif m == "global_variable":
            corrupted = ("static int _global_state = 0; // BUG: mutable global\n"
                         + corrupted)

        elif m == "missing_interface":
            corrupted = re.sub(
                r'#pragma\s+HLS\s+INTERFACE[^\n]*', '', corrupted)
            corrupted = re.sub(
                r'#pragma\s+HLS\s+interface[^\n]*', '', corrupted)

        elif m == "wrong_loop_bound":
            # Change a loop upper bound to a variable (non-static trip count)
            corrupted = re.sub(
                r'for\s*\(([^;]+);\s*([^<>=]+)(<|<=|>|>=)\s*(\d+)',
                lambda m_: f'for ({m_.group(1)}; {m_.group(2)}{m_.group(3)} unknown_bound',
                corrupted, count=1)
            corrupted = ("int unknown_bound = 999; "
                         "// BUG: non-constant loop bound\n" + corrupted)

        elif m == "double_free":
            corrupted = corrupted.replace(
                '{', '{\n    int* _p = (int*)malloc(32);\n    free(_p); free(_p); // double free\n', 1)

        elif m == "new_delete":
            corrupted = corrupted.replace(
                '{', '{\n    int* _arr = new int[256]; // BUG: new/delete not synthesizable\n    delete[] _arr;\n', 1)

        elif m == "float_compare":
            # Inject float equality comparison (unreliable in HW)
            corrupted = corrupted.replace(
                '{', '{\n    float _a = 0.1f, _b = 0.2f;\n    if (_a + _b == 0.3f) {} // BUG: float equality\n', 1)

        # --- Versal / Vitis HLS specific ---
        elif m == "remove_dataflow":
            corrupted = re.sub(
                r'#pragma\s+HLS\s+DATAFLOW[^\n]*', '', corrupted,
                flags=re.IGNORECASE)

        elif m == "break_array_partition":
            # Change ARRAY_PARTITION to a wrong factor
            corrupted = re.sub(
                r'(#pragma\s+HLS\s+ARRAY_PARTITION[^\n]*factor=)\d+',
                lambda m_: m_.group(1) + '1',  # factor=1 = no partition
                corrupted, flags=re.IGNORECASE)
            # Also try removing it entirely sometimes
            if random.random() < 0.5:
                corrupted = re.sub(
                    r'#pragma\s+HLS\s+ARRAY_PARTITION[^\n]*', '',
                    corrupted, flags=re.IGNORECASE)

        elif m == "wrong_bind_storage":
            corrupted = re.sub(
                r'#pragma\s+HLS\s+BIND_STORAGE[^\n]*', '', corrupted,
                flags=re.IGNORECASE)
            corrupted = re.sub(
                r'#pragma\s+HLS\s+BIND_OP[^\n]*', '', corrupted,
                flags=re.IGNORECASE)

        elif m == "remove_array_reshape":
            corrupted = re.sub(
                r'#pragma\s+HLS\s+ARRAY_RESHAPE[^\n]*', '', corrupted,
                flags=re.IGNORECASE)

        elif m == "bad_fifo_depth":
            # Set FIFO depth to 1 (causes deadlock in dataflow)
            corrupted = re.sub(
                r'(#pragma\s+HLS\s+STREAM[^\n]*depth=)\d+',
                lambda m_: m_.group(1) + '1',
                corrupted, flags=re.IGNORECASE)
            if 'hls::stream' in corrupted and '#pragma HLS STREAM' not in corrupted.upper():
                corrupted = corrupted.replace(
                    'hls::stream',
                    'hls::stream  /* WARNING: no depth pragma */', 1)

        elif m == "remove_inline":
            corrupted = re.sub(
                r'#pragma\s+HLS\s+INLINE[^\n]*', '', corrupted,
                flags=re.IGNORECASE)

        elif m == "partial_fix":
            # Keep ~half the pragmas, strip the rest
            pragma_lines = list(re.finditer(
                r'#pragma\s+HLS\s+[^\n]*', corrupted))
            if pragma_lines:
                to_remove = random.sample(
                    pragma_lines,
                    k=max(1, len(pragma_lines) // 2))
                for match in reversed(to_remove):  # reverse to preserve indices
                    corrupted = (corrupted[:match.start()]
                                 + corrupted[match.end():])

    return corrupted


# Varied instruction prompts so the model learns multiple debugging phrasings
DEBUG_INSTRUCTIONS = [
    "Identify and fix the HLS synthesis errors or bottlenecks in this code.",
    "This code fails Vivado HLS synthesis. Find and correct all issues.",
    "Debug this C/C++ code so it is fully synthesizable with Xilinx Vitis HLS.",
    "The following HLS code contains non-synthesizable constructs. Fix them.",
    "Analyze this code for HLS compatibility problems and provide the corrected version.",
    "This FPGA kernel has synthesis errors. Rewrite it so it passes HLS compilation.",
    "Find the hardware-incompatible patterns in this code and replace them with HLS-friendly alternatives.",
    "This Versal kernel fails to compile in Vitis HLS. Debug it and provide working code.",
    "Review this HLS/FPGA code for synthesis issues. Output the corrected version.",
    "The following code has bugs that prevent FPGA synthesis. Identify each issue and fix it.",
    "This code targets a Xilinx Versal ACAP but has errors. Rewrite it for successful synthesis.",
    "Debug the following Vitis HLS kernel. Explain what is wrong and provide the fix.",
]

# "Explain-the-bug" instructions — model must explain *why* code is broken
EXPLAIN_DEBUG_INSTRUCTIONS = [
    "Explain each HLS synthesis error in this code, then provide the corrected version.",
    "This Vitis HLS kernel has bugs. For each issue: (1) identify the problem, (2) explain why it fails synthesis, (3) show the fix.",
    "Analyze this code for FPGA synthesis errors. List each bug with an explanation of why it is not synthesizable, then output the corrected code.",
    "Debug this HLS code step by step. For each issue found, explain the root cause and how to fix it. Then provide the complete corrected code.",
    "Review this Versal kernel for synthesis compatibility. Explain what each problem is and why it matters for hardware, then give the fixed version.",
]


# ──────────────────────────────────────────────────────────
# BUG EXPLANATION GENERATOR
# ──────────────────────────────────────────────────────────
BUG_EXPLANATIONS = {
    "strip_pragmas":        "All HLS pragmas have been removed. Without pragmas, the HLS tool cannot apply optimizations like pipelining, unrolling, or array partitioning, resulting in poor performance or synthesis failure.",
    "revert_types":         "HLS-specific fixed-point types (ap_int, ap_uint, ap_fixed) have been replaced with standard C types (int, unsigned int, float). This loses bit-width control and can cause incorrect hardware sizing.",
    "inject_malloc":        "Dynamic memory allocation (malloc) is not synthesizable in HLS. Hardware requires statically-known memory sizes at compile time.",
    "inject_recursion":     "Recursion is not supported in HLS synthesis. Hardware cannot dynamically allocate stack frames. Use iterative constructs instead.",
    "swap_stream":          "hls::stream has been replaced with std::queue, which is not synthesizable. hls::stream maps to hardware FIFOs while std::queue requires dynamic memory.",
    "remove_pipeline":      "Pipeline pragmas have been removed. Without pipelining, loops execute sequentially, dramatically reducing throughput.",
    "add_pointer_cast":     "Non-synthesizable pointer casts to arbitrary addresses are present. HLS cannot map arbitrary memory addresses to hardware.",
    "variable_length_array":"Variable-length arrays (VLAs) are not supported in HLS. Array sizes must be compile-time constants for hardware synthesis.",
    "global_variable":      "Mutable global/static variables may cause synthesis issues as they imply persistent state across function calls that is hard to map to hardware.",
    "missing_interface":    "HLS INTERFACE pragmas have been removed. Without interface specification, the tool cannot properly generate RTL port protocols.",
    "wrong_loop_bound":     "A loop has a non-constant trip count. HLS needs compile-time-known loop bounds for pipeline scheduling and resource estimation.",
    "double_free":          "Double free of dynamically allocated memory. Dynamic allocation itself is not synthesizable, and double free is undefined behavior.",
    "new_delete":           "C++ new/delete operators are not synthesizable. Use statically allocated arrays instead.",
    "float_compare":        "Floating-point equality comparison is unreliable in hardware due to rounding. Use epsilon-based comparison or fixed-point types.",
    "remove_dataflow":      "DATAFLOW pragma removed. Without dataflow, functions execute sequentially instead of in a pipelined, concurrent fashion.",
    "break_array_partition": "ARRAY_PARTITION factor set to 1 or removed entirely. Without proper partitioning, memory bandwidth becomes a bottleneck.",
    "wrong_bind_storage":   "BIND_STORAGE/BIND_OP pragmas removed. Without these, the tool may use suboptimal storage or operator implementations.",
    "remove_array_reshape": "ARRAY_RESHAPE pragma removed. Array reshape combines partitioning with width adjustment for optimal memory port usage.",
    "bad_fifo_depth":       "FIFO/stream depth set to 1 or missing. Insufficient FIFO depth in dataflow regions causes deadlocks or throughput degradation.",
    "remove_inline":        "INLINE pragma removed. Without inlining, function call overhead persists and cross-function optimizations are blocked.",
    "partial_fix":          "Some HLS pragmas have been removed while others remain. The code has an incomplete optimization strategy.",
}

def generate_explanation(modes_used: list, fixed_code: str) -> str:
    """Build an 'explain + fix' output string for the explain-the-bug samples."""
    lines = ["## Issues Found\n"]
    for i, mode in enumerate(modes_used, 1):
        explanation = BUG_EXPLANATIONS.get(mode, "Unknown synthesis issue detected.")
        lines.append(f"{i}. **{mode.replace('_', ' ').title()}**: {explanation}\n")
    lines.append("\n## Corrected Code\n")
    lines.append(f"```cpp\n{fixed_code}\n```")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# GIT BUGFIX MINER — extract real bug→fix pairs from repos
# ──────────────────────────────────────────────────────────
# Repos that are known to be HLS-related — relax keyword filter for these
_HLS_REPOS = {'Vitis_Accel_Examples', 'Vitis_Libraries', 'Vitis-Tutorials',
              'VitisHLS', 'CHStone', 'MachSuite', 'HLSyn', 'ForgeHLS',
              'heterocl', 'soda-opt', 'Merlin-UCLA', 'XRT', 'AutoBridge',
              'Vitis-AI', 'Odyssey', 'OpenFPGA', 'mlir-aie',
              'vtr-verilog-to-routing'}

def mine_git_bugfixes(repo_paths: list) -> list:
    """
    Walk git history of cloned HLS repos, find commits that modified
    .c/.cpp/.h/.hpp files, and extract (before, after) pairs as real
    debugging / improvement data.
    """
    samples = []
    # Broad set of keywords for HLS relevance
    hls_keywords = [
        'pragma', 'hls', 'pipeline', 'unroll', 'partition', 'dataflow',
        'interface', 'stream', 'ap_int', 'ap_uint', 'ap_fixed', 'ap_axi',
        'hls_stream', 'axis', 'kernel', 'vitis', 'vivado', 'fpga',
        'axi', 'xf_', 'xcl_', 'ocl', 'cl_mem', 'hbm', 'ddr',
        'bind_storage', 'bind_op', 'array_reshape', 'loop_tripcount',
        'inline', 'fifo_depth', 'top', 'synthesis', 'cosim',
        '#include "hls_', '#include <hls_', 'hlslib',
    ]

    for repo_path in repo_paths:
        if not os.path.isdir(os.path.join(repo_path, '.git')):
            continue
        repo_name = os.path.basename(repo_path)
        is_hls_repo = repo_name in _HLS_REPOS
        print(f"  Mining {repo_name} {'(HLS-repo)' if is_hls_repo else ''} ...")

        try:
            result = subprocess.run(
                ['git', 'log', '--all', '--pretty=format:%H', '--diff-filter=M',
                 '--', '*.c', '*.cpp', '*.h', '*.hpp'],
                cwd=repo_path, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=60)
            if result.returncode != 0:
                continue
            commits = result.stdout.strip().split('\n')
            commits = [c for c in commits if c][:500]
        except Exception:
            continue

        repo_count = 0
        for commit_hash in commits:
            try:
                diff_result = subprocess.run(
                    ['git', 'diff', '--name-only', f'{commit_hash}~1', commit_hash,
                     '--', '*.c', '*.cpp', '*.h', '*.hpp'],
                    cwd=repo_path, capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=15)
                if diff_result.returncode != 0:
                    continue
                changed_files = [f for f in diff_result.stdout.strip().split('\n') if f]
            except Exception:
                continue

            for cfile in changed_files[:10]:
                try:
                    before = subprocess.run(
                        ['git', 'show', f'{commit_hash}~1:{cfile}'],
                        cwd=repo_path, capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=10)
                    after = subprocess.run(
                        ['git', 'show', f'{commit_hash}:{cfile}'],
                        cwd=repo_path, capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=10)

                    if before.returncode != 0 or after.returncode != 0:
                        continue

                    before_code = before.stdout
                    after_code = after.stdout

                    # Skip if both are very short
                    if len(before_code) < 30 or len(after_code) < 30:
                        continue
                    # Skip if content is truly identical
                    if before_code == after_code:
                        continue

                    # Keyword check — relaxed for known HLS repos
                    if not is_hls_repo:
                        combined = (before_code + after_code).lower()
                        if not any(kw in combined for kw in hls_keywords):
                            continue

                    # Truncate very large files to 8K chars
                    before_code = before_code[:8192]
                    after_code = after_code[:8192]

                    samples.append({
                        "instruction": random.choice(DEBUG_INSTRUCTIONS),
                        "input": before_code,
                        "output": after_code,
                        "_source": f"git:{repo_name}/{cfile}@{commit_hash[:8]}",
                    })
                    repo_count += 1
                except Exception:
                    continue

        print(f"    {repo_name}: {repo_count} pairs")

    print(f"    → Mined {len(samples):,} real bugfix pairs from git history")
    return samples


# ──────────────────────────────────────────────────────────
# KNOWN REAL-WORLD HLS BUG PATTERNS
# ──────────────────────────────────────────────────────────
# Each entry is a (buggy_code, fixed_code, description) triple
# based on documented Xilinx AR / Vitis HLS known issues.
_KNOWN_ISSUES = [
    # --- Pragma / directive bugs ---
    (
        '#pragma HLS PIPELINE II=1\nvoid top(hls::stream<int>& in, hls::stream<int>& out) {\n  for (int i = 0; i < N; i++) {\n    int val = in.read();\n    for (int j = 0; j < M; j++) {\n      #pragma HLS PIPELINE II=1\n      out.write(val + j);\n    }\n  }\n}',
        'void top(hls::stream<int>& in, hls::stream<int>& out) {\n  for (int i = 0; i < N; i++) {\n    #pragma HLS PIPELINE off\n    int val = in.read();\n    for (int j = 0; j < M; j++) {\n      #pragma HLS PIPELINE II=1\n      out.write(val + j);\n    }\n  }\n}',
        'Outer loop pipeline conflicts with inner loop pipeline; disable outer pipeline.'
    ),
    (
        'void krnl(int a[1024], int b[1024]) {\n  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem\n  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem\n  for (int i = 0; i < 1024; i++)\n    b[i] = a[i] * 2;\n}',
        'void krnl(int a[1024], int b[1024]) {\n  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0\n  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1\n  for (int i = 0; i < 1024; i++)\n    b[i] = a[i] * 2;\n}',
        'Sharing a single AXI bundle for both read and write ports causes port contention; split into separate bundles.'
    ),
    (
        'void top(ap_uint<512>* mem, int n) {\n  #pragma HLS INTERFACE m_axi port=mem depth=1024\n  int local[1024];\n  #pragma HLS ARRAY_PARTITION variable=local complete\n  memcpy(local, mem, n*sizeof(int));\n  for (int i = 0; i < n; i++) local[i] += 1;\n  memcpy(mem, local, n*sizeof(int));\n}',
        'void top(ap_uint<512>* mem, int n) {\n  #pragma HLS INTERFACE m_axi port=mem depth=1024\n  int local[1024];\n  #pragma HLS ARRAY_PARTITION variable=local cyclic factor=16\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    local[i] = ((int*)mem)[i];\n  }\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    local[i] += 1;\n  }\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    ((int*)mem)[i] = local[i];\n  }\n}',
        'Complete partition of 1024-element array uses too many FF; use cyclic partition. Also memcpy prevents pipelining.'
    ),
    # --- Dataflow violations ---
    (
        'void top(hls::stream<int>& in, hls::stream<int>& out) {\n  #pragma HLS DATAFLOW\n  int buf[256];\n  read_input(in, buf);\n  process(buf, buf);  // in-place\n  write_output(buf, out);\n}',
        'void top(hls::stream<int>& in, hls::stream<int>& out) {\n  #pragma HLS DATAFLOW\n  hls::stream<int> s1, s2;\n  #pragma HLS STREAM variable=s1 depth=256\n  #pragma HLS STREAM variable=s2 depth=256\n  read_input(in, s1);\n  process(s1, s2);\n  write_output(s2, out);\n}',
        'Dataflow requires producer-consumer channels, not shared arrays. Use hls::stream between stages.'
    ),
    (
        'void top(int* in, int* out, int n) {\n  #pragma HLS DATAFLOW\n  int tmp[1024];\n  stage1(in, tmp, n);\n  stage2(tmp, out, n);\n  stage3(tmp, out, n);  // tmp read twice\n}',
        'void top(int* in, int* out, int n) {\n  #pragma HLS DATAFLOW\n  int tmp1[1024], tmp2[1024];\n  stage1(in, tmp1, n);\n  split(tmp1, tmp2, n);  // explicit fan-out\n  stage2(tmp1, out, n);\n  stage3(tmp2, out, n);\n}',
        'Dataflow violation: tmp is consumed by two stages. Insert explicit split to create separate channels.'
    ),
    # --- Stream depth / deadlock issues ---
    (
        'void top(hls::stream<pkt>& in, hls::stream<pkt>& out) {\n  #pragma HLS DATAFLOW\n  hls::stream<pkt> fifo;\n  producer(in, fifo);\n  consumer(fifo, out);\n}',
        'void top(hls::stream<pkt>& in, hls::stream<pkt>& out) {\n  #pragma HLS DATAFLOW\n  hls::stream<pkt> fifo;\n  #pragma HLS STREAM variable=fifo depth=64\n  producer(in, fifo);\n  consumer(fifo, out);\n}',
        'Missing FIFO depth on internal stream causes cosim deadlock with default depth=1.'
    ),
    (
        'void read_engine(hls::stream<ap_uint<512>>& s, ap_uint<512>* ddr, int n) {\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    s.write(ddr[i]);\n  }\n}\nvoid compute(hls::stream<ap_uint<512>>& in, hls::stream<int>& out) {\n  while (!in.empty()) {\n    ap_uint<512> w = in.read();\n    for (int j = 0; j < 16; j++)\n      out.write(w.range(j*32+31, j*32));\n  }\n}',
        'void read_engine(hls::stream<ap_uint<512>>& s, ap_uint<512>* ddr, int n) {\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    s.write(ddr[i]);\n  }\n}\nvoid compute(hls::stream<ap_uint<512>>& in, hls::stream<int>& out, int n) {\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    ap_uint<512> w = in.read();\n    for (int j = 0; j < 16; j++)\n      out.write(w.range(j*32+31, j*32));\n  }\n}',
        'Using !stream.empty() as loop condition causes non-synthesizable or non-deterministic behavior; use bounded loop.'
    ),
    # --- Incorrect interface / port issues ---
    (
        'void krnl(int* a, int* b, int size) {\n  #pragma HLS INTERFACE m_axi port=a\n  #pragma HLS INTERFACE m_axi port=b\n  for (int i = 0; i < size; i++)\n    b[i] = a[i] + 1;\n}',
        'void krnl(int* a, int* b, int size) {\n  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0\n  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1\n  #pragma HLS INTERFACE s_axilite port=a\n  #pragma HLS INTERFACE s_axilite port=b\n  #pragma HLS INTERFACE s_axilite port=size\n  #pragma HLS INTERFACE s_axilite port=return\n  for (int i = 0; i < size; i++) {\n    #pragma HLS PIPELINE II=1\n    b[i] = a[i] + 1;\n  }\n}',
        'Missing s_axilite control interfaces and offset=slave; host cannot set base addresses without them.'
    ),
    (
        'extern "C" void vadd(int* a, int* b, int* c, int n) {\n  #pragma HLS INTERFACE m_axi port=a bundle=gmem0\n  #pragma HLS INTERFACE m_axi port=b bundle=gmem0\n  #pragma HLS INTERFACE m_axi port=c bundle=gmem0\n  for (int i = 0; i < n; i++)\n    c[i] = a[i] + b[i];\n}',
        'extern "C" void vadd(int* a, int* b, int* c, int n) {\n  #pragma HLS INTERFACE m_axi port=a bundle=gmem0\n  #pragma HLS INTERFACE m_axi port=b bundle=gmem1\n  #pragma HLS INTERFACE m_axi port=c bundle=gmem2\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    c[i] = a[i] + b[i];\n  }\n}',
        'All three ports on same AXI bundle limits to one outstanding transaction; separate bundles enable parallel access.'
    ),
    # --- Type / width bugs ---
    (
        '#include "ap_int.h"\nvoid top(ap_int<8> a, ap_int<8> b, ap_int<8>& sum) {\n  sum = a + b;  // overflow at 128+128\n}',
        '#include "ap_int.h"\nvoid top(ap_int<8> a, ap_int<8> b, ap_int<9>& sum) {\n  sum = a + b;  // 9 bits avoids overflow\n}',
        'Sum of two 8-bit values needs 9 bits to avoid overflow truncation.'
    ),
    (
        '#include "ap_fixed.h"\ntypedef ap_fixed<16,8> fixed_t;\nvoid top(fixed_t x, fixed_t& y) {\n  y = x * x * x;  // intermediate overflow\n}',
        '#include "ap_fixed.h"\ntypedef ap_fixed<16,8> fixed_t;\ntypedef ap_fixed<32,16> wide_t;\nvoid top(fixed_t x, fixed_t& y) {\n  wide_t tmp = (wide_t)x * x;\n  y = (fixed_t)(tmp * x);\n}',
        'Chained multiplication overflows intermediate precision; widen intermediate type.'
    ),
    # --- Loop bound / trip-count issues ---
    (
        'void top(int a[N]) {\n  for (int i = 0; i < N; i++) {\n    if (a[i] == 0) break;  // variable trip count\n    #pragma HLS PIPELINE II=1\n    a[i] *= 2;\n  }\n}',
        'void top(int a[N]) {\n  for (int i = 0; i < N; i++) {\n    #pragma HLS PIPELINE II=1\n    #pragma HLS LOOP_TRIPCOUNT min=1 max=N\n    if (a[i] == 0) break;\n    a[i] *= 2;\n  }\n}',
        'Variable trip-count loop needs LOOP_TRIPCOUNT pragma for accurate latency estimation.'
    ),
    (
        'void top(hls::stream<int>& in, int out[1024]) {\n  int i = 0;\n  while (!in.empty()) {\n    out[i++] = in.read();\n  }\n}',
        'void top(hls::stream<int>& in, int out[1024], int n) {\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    out[i] = in.read();\n  }\n}',
        'Unbounded while loop with stream.empty() is not synthesizable; use bounded for loop with known count.'
    ),
    # --- Memory / array issues ---
    (
        'void top(int A[1024][1024]) {\n  #pragma HLS ARRAY_PARTITION variable=A complete dim=1\n  for (int i = 0; i < 1024; i++)\n    for (int j = 0; j < 1024; j++)\n      A[i][j] += 1;\n}',
        'void top(int A[1024][1024]) {\n  #pragma HLS ARRAY_PARTITION variable=A cyclic factor=4 dim=1\n  for (int i = 0; i < 1024; i++)\n    for (int j = 0; j < 1024; j++) {\n      #pragma HLS PIPELINE II=1\n      A[i][j] += 1;\n    }\n}',
        'Complete partition of dim=1 (1024 elements) creates 1024 BRAMs; use cyclic factor instead.'
    ),
    (
        'void top(int a[1024], int b[1024]) {\n  #pragma HLS BIND_STORAGE variable=a type=RAM_1P impl=BRAM\n  for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    b[i] = a[i] + a[1023-i];  // dual read\n  }\n}',
        'void top(int a[1024], int b[1024]) {\n  #pragma HLS BIND_STORAGE variable=a type=RAM_2P impl=BRAM\n  for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    b[i] = a[i] + a[1023-i];\n  }\n}',
        'Two concurrent reads from same array needs RAM_2P (dual port), not RAM_1P which limits to II=2.'
    ),
    # --- Versal / AIE specific ---
    (
        '#include "adf.h"\nclass myGraph : public adf::graph {\npublic:\n  adf::kernel k1;\n  adf::port<input> in;\n  adf::port<output> out;\n  myGraph() {\n    k1 = adf::kernel::create(filter);\n    adf::connect<>(in, k1.in[0]);\n    adf::connect<>(k1.out[0], out);\n  }\n};',
        '#include "adf.h"\nclass myGraph : public adf::graph {\npublic:\n  adf::kernel k1;\n  adf::port<input> in;\n  adf::port<output> out;\n  myGraph() {\n    k1 = adf::kernel::create(filter);\n    adf::runtime<ratio>(k1) = 0.8;\n    adf::source(k1) = "filter.cc";\n    adf::connect<adf::window<256>>(in, k1.in[0]);\n    adf::connect<adf::window<256>>(k1.out[0], out);\n  }\n};',
        'AIE graph missing runtime ratio, source file, and window size on connections.'
    ),
    (
        'void filter(input_window<int32>* in, output_window<int32>* out) {\n  for (int i = 0; i < 256; i++) {\n    int32 val = window_readincr(in);\n    window_writeincr(out, val * coeff[i]);\n  }\n}',
        'void filter(input_window<int32>* __restrict in, output_window<int32>* __restrict out) {\n  v8int32 buf = undef_v8int32();\n  v8int32 coe = undef_v8int32();\n  for (int i = 0; i < 256/8; i++)\n    chess_prepare_for_pipelining {\n    buf = window_read_v8(in); window_incr(in, 8);\n    coe = *(v8int32*)&coeff[i*8];\n    v8int32 res = mul8(buf, coe);\n    window_write_v8(out, res); window_incr(out, 8);\n  }\n}',
        'Scalar AIE kernel wastes VLIW vector unit; vectorize with v8int32 intrinsics for 8x throughput.'
    ),
    (
        'void dma_mm2s(ap_uint<128>* mem, hls::stream<ap_axiu<128,0,0,0>>& s, int n) {\n  for (int i = 0; i < n; i++) {\n    ap_axiu<128,0,0,0> pkt;\n    pkt.data = mem[i];\n    pkt.last = (i == n-1);\n    pkt.keep = -1;\n    s.write(pkt);\n  }\n}',
        'void dma_mm2s(ap_uint<128>* mem, hls::stream<ap_axiu<128,0,0,0>>& s, int n) {\n  #pragma HLS INTERFACE m_axi port=mem offset=slave bundle=gmem\n  #pragma HLS INTERFACE axis port=s\n  #pragma HLS INTERFACE s_axilite port=mem\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    ap_axiu<128,0,0,0> pkt;\n    pkt.data = mem[i];\n    pkt.last = (i == n-1);\n    pkt.keep = -1;\n    s.write(pkt);\n  }\n}',
        'PL DMA kernel missing all interface pragmas; needs m_axi, axis, s_axilite for Versal platform integration.'
    ),
    # --- Latency / performance bugs ---
    (
        'void top(int A[N][N], int B[N][N], int C[N][N]) {\n  for (int i = 0; i < N; i++)\n    for (int j = 0; j < N; j++) {\n      int sum = 0;\n      for (int k = 0; k < N; k++)\n        sum += A[i][k] * B[k][j];\n      C[i][j] = sum;\n    }\n}',
        'void top(int A[N][N], int B[N][N], int C[N][N]) {\n  #pragma HLS ARRAY_PARTITION variable=B cyclic factor=16 dim=2\n  for (int i = 0; i < N; i++)\n    for (int j = 0; j < N; j++) {\n      #pragma HLS PIPELINE II=1\n      int sum = 0;\n      for (int k = 0; k < N; k++)\n        #pragma HLS UNROLL factor=16\n        sum += A[i][k] * B[k][j];\n      C[i][j] = sum;\n    }\n}',
        'Matrix multiply without PIPELINE/UNROLL/PARTITION has II=N*N*N; partition B and unroll inner loop.'
    ),
    (
        'void top(volatile int* in, volatile int* out, int n) {\n  for (int i = 0; i < n; i++)\n    out[i] = in[i];\n}',
        'void top(volatile int* in, volatile int* out, int n) {\n  #pragma HLS INTERFACE m_axi port=in  bundle=gmem0 max_read_burst_length=64\n  #pragma HLS INTERFACE m_axi port=out bundle=gmem1 max_write_burst_length=64\n  int buf[64];\n  for (int b = 0; b < n; b += 64) {\n    int len = (b + 64 <= n) ? 64 : n - b;\n    memcpy(buf, in + b, len * sizeof(int));\n    memcpy(out + b, buf, len * sizeof(int));\n  }\n}',
        'Single-element memory access has no burst; use local buffer + memcpy for burst inference.'
    ),
    # --- Synthesis / compilation errors ---
    (
        '#include <cstdlib>\nvoid top(int* a, int n) {\n  int* tmp = (int*)malloc(n * sizeof(int));\n  for (int i = 0; i < n; i++) tmp[i] = a[i];\n  for (int i = 0; i < n; i++) a[i] = tmp[n-1-i];\n  free(tmp);\n}',
        'void top(int a[1024]) {\n  int tmp[1024];\n  for (int i = 0; i < 1024; i++)\n    #pragma HLS PIPELINE II=1\n    tmp[i] = a[i];\n  for (int i = 0; i < 1024; i++)\n    #pragma HLS PIPELINE II=1\n    a[i] = tmp[1023-i];\n}',
        'Dynamic memory allocation (malloc/free) is not synthesizable in HLS; use fixed-size arrays.'
    ),
    (
        '#include <vector>\nvoid top(std::vector<int>& data) {\n  std::sort(data.begin(), data.end());\n}',
        'void top(int data[1024], int n) {\n  // Bitonic sort — synthesizable in HLS\n  for (int k = 2; k <= n; k <<= 1) {\n    for (int j = k >> 1; j > 0; j >>= 1) {\n      #pragma HLS PIPELINE II=1\n      for (int i = 0; i < n; i++) {\n        int ij = i ^ j;\n        if (ij > i) {\n          if ((i & k) == 0 && data[i] > data[ij])\n            { int t = data[i]; data[i] = data[ij]; data[ij] = t; }\n          else if ((i & k) != 0 && data[i] < data[ij])\n            { int t = data[i]; data[i] = data[ij]; data[ij] = t; }\n        }\n      }\n    }\n  }\n}',
        'std::vector and std::sort are not synthesizable; use fixed array and bitonic sort.'
    ),
    (
        'void top(float a[256], float b[256], float c[256]) {\n  for (int i = 0; i < 256; i++)\n    c[i] = a[i] / b[i];\n}',
        'void top(float a[256], float b[256], float c[256]) {\n  #pragma HLS BIND_OP variable=c op=fdiv impl=fabric latency=12\n  for (int i = 0; i < 256; i++) {\n    #pragma HLS PIPELINE II=1\n    c[i] = a[i] / b[i];\n  }\n}',
        'Floating-point division without BIND_OP uses default impl with high II; specify latency for pipelining.'
    ),
    # --- Common Vitis platform issues ---
    (
        'extern "C" {\nvoid krnl_vadd(int* a, int* b, int* c, int n) {\n  for (int i = 0; i < n; i++)\n    c[i] = a[i] + b[i];\n}\n}',
        'extern "C" {\nvoid krnl_vadd(int* a, int* b, int* c, int n) {\n  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0\n  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1\n  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2\n  #pragma HLS INTERFACE s_axilite port=a\n  #pragma HLS INTERFACE s_axilite port=b\n  #pragma HLS INTERFACE s_axilite port=c\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n\n  int buf_a[4096], buf_b[4096], buf_c[4096];\n  for (int i = 0; i < n; i += 4096) {\n    int chunk = ((i + 4096) < n) ? 4096 : (n - i);\n    memcpy(buf_a, a + i, chunk * sizeof(int));\n    memcpy(buf_b, b + i, chunk * sizeof(int));\n    for (int j = 0; j < chunk; j++)\n      #pragma HLS PIPELINE II=1\n      buf_c[j] = buf_a[j] + buf_b[j];\n    memcpy(c + i, buf_c, chunk * sizeof(int));\n  }\n}\n}',
        'Vitis kernel needs interface pragmas, local buffers for burst, and pipeline.'
    ),
    (
        'void krnl(hls::stream<ap_axiu<32,0,0,0>>& in,\n           hls::stream<ap_axiu<32,0,0,0>>& out, int n) {\n  for (int i = 0; i < n; i++) {\n    auto pkt = in.read();\n    pkt.data = pkt.data + 1;\n    out.write(pkt);\n  }\n}',
        'void krnl(hls::stream<ap_axiu<32,0,0,0>>& in,\n           hls::stream<ap_axiu<32,0,0,0>>& out, int n) {\n  #pragma HLS INTERFACE axis port=in\n  #pragma HLS INTERFACE axis port=out\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    ap_axiu<32,0,0,0> pkt = in.read();\n    pkt.data = pkt.data + 1;\n    if (i == n - 1) pkt.last = 1;\n    out.write(pkt);\n  }\n}',
        'AXI-Stream kernel missing axis/s_axilite pragmas, PIPELINE, and TLAST signal management.'
    ),
    # --- Reset / initialization bugs ---
    (
        'static int state = 0;\nvoid top(int in_val, int& out_val) {\n  #pragma HLS PIPELINE II=1\n  state += in_val;\n  out_val = state;\n}',
        'static int state = 0;\nvoid top(int in_val, int& out_val, bool reset) {\n  #pragma HLS PIPELINE II=1\n  #pragma HLS RESET variable=state\n  if (reset) state = 0;\n  else state += in_val;\n  out_val = state;\n}',
        'Static variable without reset pragma cannot be re-initialized; add HLS RESET and reset port.'
    ),
    # --- Vivado HLS vs Vitis HLS migration ---
    (
        '#pragma HLS resource variable=a core=RAM_2P_BRAM\n#pragma HLS LOOP_FLATTEN off\nvoid top(int a[1024]) {\n  for (int i = 0; i < 1024; i++)\n    a[i] *= 2;\n}',
        '#pragma HLS BIND_STORAGE variable=a type=RAM_2P impl=BRAM\n#pragma HLS LOOP_FLATTEN off\nvoid top(int a[1024]) {\n  for (int i = 0; i < 1024; i++)\n    #pragma HLS PIPELINE II=1\n    a[i] *= 2;\n}',
        'Deprecated Vivado HLS pragma "resource" replaced with Vitis HLS "BIND_STORAGE" syntax.'
    ),
    (
        '#pragma HLS RESOURCE variable=mul core=Mul_LUT\nvoid top(short a, short b, int& c) { c = a * b; }',
        '#pragma HLS BIND_OP variable=mul op=mul impl=fabric\nvoid top(short a, short b, int& c) {\n  #pragma HLS BIND_OP variable=c op=mul impl=fabric\n  c = a * b;\n}',
        'Vivado HLS RESOURCE pragma with core= is deprecated; use BIND_OP with op= and impl= in Vitis.'
    ),
    # --- Additional real-world patterns ---
    (
        'void top(int in[1024], int out[1024]) {\n  #pragma HLS DATAFLOW\n  int buf[1024];\n  for (int i = 0; i < 1024; i++) buf[i] = in[i];\n  for (int i = 0; i < 1024; i++) buf[i] = buf[i] * 2;\n  for (int i = 0; i < 1024; i++) out[i] = buf[i];\n}',
        'void top(int in[1024], int out[1024]) {\n  #pragma HLS DATAFLOW\n  hls::stream<int> s1, s2;\n  #pragma HLS STREAM variable=s1 depth=16\n  #pragma HLS STREAM variable=s2 depth=16\n  load: for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    s1.write(in[i]);\n  }\n  compute: for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    s2.write(s1.read() * 2);\n  }\n  store: for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    out[i] = s2.read();\n  }\n}',
        'Dataflow with shared array buf violates single-producer single-consumer rule; convert to stream channels.'
    ),
    (
        'void top(hls::stream<int>& in, hls::stream<int>& out) {\n  int acc = 0;\n  LOOP: for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    acc += in.read();\n  }\n  out.write(acc);\n}',
        'void top(hls::stream<int>& in, hls::stream<int>& out) {\n  #pragma HLS INTERFACE axis port=in\n  #pragma HLS INTERFACE axis port=out\n  #pragma HLS INTERFACE ap_ctrl_none port=return\n  int acc = 0;\n  LOOP: for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    acc += in.read();\n  }\n  out.write(acc);\n}',
        'Streaming kernel missing axis interface pragmas; host cannot connect AXI-Stream ports without them.'
    ),
    (
        'void conv2d(int img[H][W], int kernel[K][K], int out[H][W]) {\n  for (int r = 0; r < H; r++)\n    for (int c = 0; c < W; c++) {\n      int sum = 0;\n      for (int kr = 0; kr < K; kr++)\n        for (int kc = 0; kc < K; kc++)\n          sum += img[r+kr][c+kc] * kernel[kr][kc];\n      out[r][c] = sum;\n    }\n}',
        'void conv2d(int img[H][W], int kernel[K][K], int out[H][W]) {\n  #pragma HLS ARRAY_PARTITION variable=kernel complete\n  int linebuf[K][W];\n  #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1\n  for (int r = 0; r < H; r++)\n    for (int c = 0; c < W; c++) {\n      #pragma HLS PIPELINE II=1\n      // Shift line buffer\n      for (int kr = 0; kr < K-1; kr++)\n        linebuf[kr][c] = linebuf[kr+1][c];\n      linebuf[K-1][c] = img[r][c];\n      if (r >= K-1 && c >= K-1) {\n        int sum = 0;\n        for (int kr = 0; kr < K; kr++)\n          #pragma HLS UNROLL\n          for (int kc = 0; kc < K; kc++)\n            #pragma HLS UNROLL\n            sum += linebuf[kr][c-K+1+kc] * kernel[kr][kc];\n        out[r-K+1][c-K+1] = sum;\n      }\n    }\n}',
        'Naive 2D convolution has random memory access pattern; use line buffer for streaming access with II=1.'
    ),
    (
        'void top(ap_uint<512>* mem, int n) {\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    ap_uint<512> word = mem[i];\n    // process 16 ints packed in 512 bits\n    for (int j = 0; j < 16; j++) {\n      int val = word.range(j*32+31, j*32);\n      // ... process val\n    }\n  }\n}',
        'void top(ap_uint<512>* mem, int n) {\n  #pragma HLS INTERFACE m_axi port=mem offset=slave bundle=gmem max_read_burst_length=64\n  #pragma HLS INTERFACE s_axilite port=mem\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    ap_uint<512> word = mem[i];\n    for (int j = 0; j < 16; j++) {\n      #pragma HLS UNROLL\n      int val = word.range(j*32+31, j*32);\n      // ... process val\n    }\n  }\n}',
        'Wide-bus access pattern missing interface pragmas and inner loop UNROLL; prevents burst and serializes extraction.'
    ),
    (
        '#include "hls_stream.h"\nvoid split(hls::stream<int>& in, hls::stream<int>& out1, hls::stream<int>& out2, int n) {\n  for (int i = 0; i < n; i++) {\n    int val = in.read();\n    out1.write(val);\n    out2.write(val);\n  }\n}\nvoid top(hls::stream<int>& in, hls::stream<int>& out1, hls::stream<int>& out2) {\n  #pragma HLS DATAFLOW\n  hls::stream<int> mid;\n  split(in, out1, out2, 1024);\n}',
        '#include "hls_stream.h"\nvoid split(hls::stream<int>& in, hls::stream<int>& out1, hls::stream<int>& out2, int n) {\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    int val = in.read();\n    out1.write(val);\n    out2.write(val);\n  }\n}\nvoid top(hls::stream<int>& in, hls::stream<int>& out1, hls::stream<int>& out2) {\n  #pragma HLS DATAFLOW\n  hls::stream<int> mid;\n  #pragma HLS STREAM variable=mid depth=32\n  split(in, out1, out2, 1024);\n}',
        'Split function missing PIPELINE; internal stream missing depth pragma causes potential backpressure deadlock.'
    ),
    (
        'void top(int A[N][N], int B[N][N], int C[N][N]) {\n  for (int i = 0; i < N; i++) {\n    for (int j = 0; j < N; j++) {\n      #pragma HLS PIPELINE II=1\n      C[i][j] = 0;\n      for (int k = 0; k < N; k++)\n        C[i][j] += A[i][k] * B[k][j];\n    }\n  }\n}',
        'void top(int A[N][N], int B[N][N], int C[N][N]) {\n  #pragma HLS ARRAY_PARTITION variable=A cyclic factor=8 dim=2\n  #pragma HLS ARRAY_PARTITION variable=B cyclic factor=8 dim=1\n  for (int i = 0; i < N; i++) {\n    for (int j = 0; j < N; j++) {\n      #pragma HLS PIPELINE II=1\n      int sum = 0;\n      for (int k = 0; k < N; k++)\n        #pragma HLS UNROLL factor=8\n        sum += A[i][k] * B[k][j];\n      C[i][j] = sum;\n    }\n  }\n}',
        'Matrix multiply accumulates directly into C array causing read-after-write; use local sum variable. Also needs partition+unroll for parallelism.'
    ),
    (
        'void histogram(int data[N], int hist[256]) {\n  for (int i = 0; i < N; i++) {\n    #pragma HLS PIPELINE II=1\n    hist[data[i]]++;\n  }\n}',
        'void histogram(int data[N], int hist[256]) {\n  #pragma HLS ARRAY_PARTITION variable=hist complete\n  int local_hist[256];\n  #pragma HLS ARRAY_PARTITION variable=local_hist complete\n  for (int i = 0; i < 256; i++)\n    #pragma HLS UNROLL\n    local_hist[i] = 0;\n  for (int i = 0; i < N; i++) {\n    #pragma HLS PIPELINE II=1\n    #pragma HLS DEPENDENCE variable=local_hist inter false\n    local_hist[data[i]]++;\n  }\n  for (int i = 0; i < 256; i++)\n    #pragma HLS UNROLL\n    hist[i] = local_hist[i];\n}',
        'Histogram has read-after-write dependency on hist array giving II=2+; use local copy with DEPENDENCE false pragma.'
    ),
    (
        'void top(float input[256], float weights[256][128], float output[128]) {\n  for (int j = 0; j < 128; j++) {\n    float sum = 0;\n    for (int i = 0; i < 256; i++)\n      sum += input[i] * weights[i][j];\n    output[j] = sum;\n  }\n}',
        'void top(float input[256], float weights[256][128], float output[128]) {\n  #pragma HLS ARRAY_PARTITION variable=input cyclic factor=4\n  #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=4 dim=1\n  for (int j = 0; j < 128; j++) {\n    #pragma HLS PIPELINE II=1\n    float sum = 0;\n    for (int i = 0; i < 256; i++)\n      #pragma HLS UNROLL factor=4\n      sum += input[i] * weights[i][j];\n    output[j] = sum;\n  }\n}',
        'Dense layer without partitioning or unrolling is fully sequential; partition inputs and unroll dot product.'
    ),
    (
        'void top(hls::stream<axis_t>& in, hls::stream<axis_t>& out) {\n  axis_t pkt;\n  do {\n    pkt = in.read();\n    pkt.data += 1;\n    out.write(pkt);\n  } while (!pkt.last);\n}',
        'void top(hls::stream<axis_t>& in, hls::stream<axis_t>& out) {\n  #pragma HLS INTERFACE axis port=in\n  #pragma HLS INTERFACE axis port=out\n  #pragma HLS INTERFACE ap_ctrl_none port=return\n  bool last = false;\n  while (!last) {\n    #pragma HLS PIPELINE II=1\n    axis_t pkt = in.read();\n    last = pkt.last;\n    pkt.data += 1;\n    out.write(pkt);\n  }\n}',
        'Do-while with TLAST check processes one extra packet; restructure as while-not-last. Add interface pragmas and ap_ctrl_none for free-running kernel.'
    ),
    (
        'void top(int A[100], int B[100]) {\n  #pragma HLS ALLOCATION instances=mul limit=1 function\n  for (int i = 0; i < 100; i++) {\n    #pragma HLS PIPELINE II=1\n    B[i] = A[i] * A[i] * A[i];  // needs 2 muls\n  }\n}',
        'void top(int A[100], int B[100]) {\n  for (int i = 0; i < 100; i++) {\n    #pragma HLS PIPELINE II=1\n    int sq = A[i] * A[i];\n    B[i] = sq * A[i];\n  }\n}',
        'ALLOCATION limiting to 1 multiplier conflicts with II=1 when 2 muls needed per iteration; remove allocation or restructure.'
    ),
    (
        'void aes_encrypt(ap_uint<128> key, ap_uint<128> plaintext, ap_uint<128>& ciphertext) {\n  ap_uint<128> state = plaintext ^ key;\n  for (int round = 1; round < 10; round++) {\n    state = sub_bytes(state);\n    state = shift_rows(state);\n    state = mix_columns(state);\n    state = add_round_key(state, round_keys[round]);\n  }\n  state = sub_bytes(state);\n  state = shift_rows(state);\n  ciphertext = state ^ round_keys[10];\n}',
        'void aes_encrypt(ap_uint<128> key, ap_uint<128> plaintext, ap_uint<128>& ciphertext) {\n  #pragma HLS PIPELINE II=1\n  #pragma HLS ARRAY_PARTITION variable=round_keys complete\n  #pragma HLS ARRAY_PARTITION variable=sbox complete\n  ap_uint<128> state = plaintext ^ key;\n  for (int round = 1; round < 10; round++) {\n    #pragma HLS UNROLL\n    state = sub_bytes(state);\n    state = shift_rows(state);\n    state = mix_columns(state);\n    state = add_round_key(state, round_keys[round]);\n  }\n  state = sub_bytes(state);\n  state = shift_rows(state);\n  ciphertext = state ^ round_keys[10];\n}',
        'AES without full unroll and partition serializes all 10 rounds; unroll rounds and partition lookup tables for single-cycle throughput.'
    ),
    (
        'void top(int* input, int* output, int n) {\n  for (int i = 1; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    output[i] = input[i] + output[i-1]; // feedback\n  }\n}',
        'void top(int* input, int* output, int n) {\n  #pragma HLS INTERFACE m_axi port=input bundle=gmem0\n  #pragma HLS INTERFACE m_axi port=output bundle=gmem1\n  int local_in[4096], local_out[4096];\n  for (int b = 0; b < n; b += 4096) {\n    int chunk = (b+4096<n) ? 4096 : n-b;\n    memcpy(local_in, input+b, chunk*sizeof(int));\n    local_out[0] = local_in[0] + (b>0 ? local_out[chunk-1] : 0);\n    for (int i = 1; i < chunk; i++) {\n      #pragma HLS PIPELINE II=1\n      #pragma HLS DEPENDENCE variable=local_out inter distance=1 true\n      local_out[i] = local_in[i] + local_out[i-1];\n    }\n    memcpy(output+b, local_out, chunk*sizeof(int));\n  }\n}',
        'Prefix-sum on external memory has II=N due to pointer aliasing; tile with local buffers and use DEPENDENCE pragma for carried dependency.'
    ),
    (
        'void top(hls::stream<int>& in, hls::stream<int>& out, int n) {\n  int buf[MAX_N];\n  for (int i = 0; i < n; i++) buf[i] = in.read();\n  // reverse\n  for (int i = 0; i < n; i++) out.write(buf[n-1-i]);\n}',
        'void top(hls::stream<int>& in, hls::stream<int>& out, int n) {\n  #pragma HLS INTERFACE axis port=in\n  #pragma HLS INTERFACE axis port=out\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n  int buf[MAX_N];\n  #pragma HLS BIND_STORAGE variable=buf type=RAM_2P impl=BRAM\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    buf[i] = in.read();\n  }\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    out.write(buf[n-1-i]);\n  }\n}',
        'Stream reverse kernel missing interface pragmas, PIPELINE, and BRAM binding for dual-port access pattern.'
    ),
    (
        'typedef struct { int x; int y; int z; } point_t;\nvoid top(point_t pts[1024], int dists[1024]) {\n  for (int i = 0; i < 1024; i++)\n    dists[i] = pts[i].x*pts[i].x + pts[i].y*pts[i].y + pts[i].z*pts[i].z;\n}',
        'typedef struct { int x; int y; int z; } point_t;\nvoid top(point_t pts[1024], int dists[1024]) {\n  #pragma HLS DATA_PACK variable=pts\n  for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    point_t p = pts[i];\n    dists[i] = p.x*p.x + p.y*p.y + p.z*p.z;\n  }\n}',
        'Struct array without DATA_PACK is split into separate arrays by HLS; pack to enable single-port access per struct.'
    ),
    (
        'void fir(int input[N], int output[N], int coeffs[TAPS]) {\n  static int shift_reg[TAPS];\n  for (int i = 0; i < N; i++) {\n    #pragma HLS PIPELINE II=1\n    int acc = 0;\n    for (int j = TAPS-1; j > 0; j--)\n      shift_reg[j] = shift_reg[j-1];\n    shift_reg[0] = input[i];\n    for (int j = 0; j < TAPS; j++)\n      acc += shift_reg[j] * coeffs[j];\n    output[i] = acc;\n  }\n}',
        'void fir(int input[N], int output[N], int coeffs[TAPS]) {\n  #pragma HLS ARRAY_PARTITION variable=coeffs complete\n  static int shift_reg[TAPS];\n  #pragma HLS ARRAY_PARTITION variable=shift_reg complete\n  for (int i = 0; i < N; i++) {\n    #pragma HLS PIPELINE II=1\n    int acc = 0;\n    shift_loop: for (int j = TAPS-1; j > 0; j--)\n      #pragma HLS UNROLL\n      shift_reg[j] = shift_reg[j-1];\n    shift_reg[0] = input[i];\n    mac_loop: for (int j = 0; j < TAPS; j++)\n      #pragma HLS UNROLL\n      acc += shift_reg[j] * coeffs[j];\n    output[i] = acc;\n  }\n}',
        'FIR filter shift register and MAC without partition/unroll cannot achieve II=1; fully partition and unroll inner loops.'
    ),
    (
        'void krnl(int* a, int* b, int n) {\n  #pragma HLS INTERFACE m_axi port=a bundle=gmem\n  #pragma HLS INTERFACE m_axi port=b bundle=gmem\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    b[i] = a[i] * 2;\n  }\n}',
        'void krnl(int* a, int* b, int n) {\n  #pragma HLS INTERFACE m_axi port=a bundle=gmem0 max_read_burst_length=256\n  #pragma HLS INTERFACE m_axi port=b bundle=gmem1 max_write_burst_length=256\n  #pragma HLS INTERFACE s_axilite port=a\n  #pragma HLS INTERFACE s_axilite port=b\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n  int buf_a[256], buf_b[256];\n  for (int offset = 0; offset < n; offset += 256) {\n    int len = (offset+256 < n) ? 256 : n - offset;\n    memcpy(buf_a, a+offset, len*sizeof(int));\n    for (int i = 0; i < len; i++)\n      #pragma HLS PIPELINE II=1\n      buf_b[i] = buf_a[i] * 2;\n    memcpy(b+offset, buf_b, len*sizeof(int));\n  }\n}',
        'Read and write on same gmem bundle serialize; split bundles, add burst hints, tile with local buffers.'
    ),
    (
        'extern "C" void vadd(ap_uint<256>* in1, ap_uint<256>* in2, ap_uint<256>* out, int n) {\n  for (int i = 0; i < n; i++) {\n    out[i] = in1[i] + in2[i]; // 256-bit add\n  }\n}',
        'extern "C" void vadd(ap_uint<256>* in1, ap_uint<256>* in2, ap_uint<256>* out, int n) {\n  #pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 offset=slave\n  #pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 offset=slave\n  #pragma HLS INTERFACE m_axi port=out bundle=gmem2 offset=slave\n  #pragma HLS INTERFACE s_axilite port=in1\n  #pragma HLS INTERFACE s_axilite port=in2\n  #pragma HLS INTERFACE s_axilite port=out\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    out[i] = in1[i] + in2[i];\n  }\n}',
        'Wide-data kernel entirely missing pragmas; needs per-port bundles, s_axilite, and PIPELINE.'
    ),
    (
        'void top(int A[1024]) {\n  #pragma HLS ARRAY_PARTITION variable=A cyclic factor=4\n  for (int i = 0; i < 1024; i += 4) {\n    #pragma HLS PIPELINE II=1\n    for (int j = 0; j < 4; j++)\n      A[i+j] = A[i+j] + 1;\n  }\n}',
        'void top(int A[1024]) {\n  #pragma HLS ARRAY_PARTITION variable=A cyclic factor=4\n  for (int i = 0; i < 1024; i += 4) {\n    #pragma HLS PIPELINE II=1\n    for (int j = 0; j < 4; j++)\n      #pragma HLS UNROLL\n      A[i+j] = A[i+j] + 1;\n  }\n}',
        'Inner loop accessing partitioned array not unrolled; partition is useless without matching UNROLL.'
    ),
    (
        'void top(hls::stream<int>& in, int table[256], hls::stream<int>& out) {\n  #pragma HLS BIND_STORAGE variable=table type=ROM_1P impl=LUTRAM\n  for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    int idx = in.read();\n    int val1 = table[idx & 0xFF];\n    int val2 = table[(idx >> 8) & 0xFF];\n    out.write(val1 + val2);\n  }\n}',
        'void top(hls::stream<int>& in, int table[256], hls::stream<int>& out) {\n  #pragma HLS BIND_STORAGE variable=table type=ROM_2P impl=LUTRAM\n  for (int i = 0; i < 1024; i++) {\n    #pragma HLS PIPELINE II=1\n    int idx = in.read();\n    int val1 = table[idx & 0xFF];\n    int val2 = table[(idx >> 8) & 0xFF];\n    out.write(val1 + val2);\n  }\n}',
        'Two reads per cycle from ROM_1P gives II=2; use ROM_2P for dual-port lookup.'
    ),
    (
        'void dma_s2mm(hls::stream<ap_axiu<64,0,0,0>>& s, ap_uint<64>* mem, int n) {\n  for (int i = 0; i < n; i++) {\n    mem[i] = s.read().data;\n  }\n}',
        'void dma_s2mm(hls::stream<ap_axiu<64,0,0,0>>& s, ap_uint<64>* mem, int n) {\n  #pragma HLS INTERFACE axis port=s\n  #pragma HLS INTERFACE m_axi port=mem offset=slave bundle=gmem max_write_burst_length=64\n  #pragma HLS INTERFACE s_axilite port=mem\n  #pragma HLS INTERFACE s_axilite port=n\n  #pragma HLS INTERFACE s_axilite port=return\n  ap_uint<64> buf[64];\n  for (int offset = 0; offset < n; offset += 64) {\n    int len = (offset+64<n) ? 64 : n-offset;\n    for (int i = 0; i < len; i++)\n      #pragma HLS PIPELINE II=1\n      buf[i] = s.read().data;\n    memcpy(mem+offset, buf, len*sizeof(ap_uint<64>));\n  }\n}',
        'S2MM DMA missing all pragmas and writing one element at a time; add interfaces, tile with buffer for write burst.'
    ),
    (
        'void top(float a[N], float b[N], float& dot) {\n  dot = 0;\n  for (int i = 0; i < N; i++)\n    dot += a[i] * b[i];\n}',
        'void top(float a[N], float b[N], float& dot) {\n  #pragma HLS ARRAY_PARTITION variable=a cyclic factor=8\n  #pragma HLS ARRAY_PARTITION variable=b cyclic factor=8\n  float partial[8] = {0};\n  #pragma HLS ARRAY_PARTITION variable=partial complete\n  for (int i = 0; i < N; i += 8) {\n    #pragma HLS PIPELINE II=1\n    for (int j = 0; j < 8; j++)\n      #pragma HLS UNROLL\n      partial[j] += a[i+j] * b[i+j];\n  }\n  float sum = 0;\n  for (int j = 0; j < 8; j++)\n    #pragma HLS UNROLL\n    sum += partial[j];\n  dot = sum;\n}',
        'Dot product with single accumulator has loop-carried dependency at float latency (~7 cycles); use partial sums with factor=8 to hide FP latency.'
    ),
    (
        'void relu(float data[1024]) {\n  for (int i = 0; i < 1024; i++)\n    if (data[i] < 0) data[i] = 0;\n}',
        'void relu(hls::stream<float>& in, hls::stream<float>& out, int n) {\n  for (int i = 0; i < n; i++) {\n    #pragma HLS PIPELINE II=1\n    float val = in.read();\n    out.write(val < 0 ? 0.0f : val);\n  }\n}',
        'In-place ReLU on array prevents dataflow integration; convert to stream-in/stream-out for composability.'
    ),
]

def generate_known_issue_samples() -> list:
    """Create training samples from known real-world HLS bugs."""
    samples = []
    debug_instr = DEBUG_INSTRUCTIONS + EXPLAIN_DEBUG_INSTRUCTIONS
    for buggy, fixed, description in _KNOWN_ISSUES:
        # Debug sample: buggy → fixed
        samples.append({
            "instruction": random.choice(DEBUG_INSTRUCTIONS),
            "input": buggy,
            "output": fixed,
        })
        # Explain sample: buggy → explanation + fixed
        explanation = (f"## Bug Analysis\n\n"
                       f"**Issue:** {description}\n\n"
                       f"## Corrected Code\n\n```cpp\n{fixed}\n```")
        samples.append({
            "instruction": random.choice(EXPLAIN_DEBUG_INSTRUCTIONS),
            "input": buggy,
            "output": explanation,
        })
    # Multiply with small variations (swap instruction prompts)
    extra = []
    for _ in range(9):  # 10x total
        for buggy, fixed, description in _KNOWN_ISSUES:
            samples.append({
                "instruction": random.choice(debug_instr),
                "input": buggy,
                "output": fixed,
            })
    print(f"  → Generated {len(samples):,} known-issue samples from {len(_KNOWN_ISSUES)} patterns")
    return samples


# ──────────────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────────────
def dedup_by_input(train: list, eval_set: list) -> tuple:
    """Remove any eval samples whose input appears in the train set."""
    train_hashes = set()
    for entry in train:
        h = hashlib.md5(entry.get("input", "").encode('utf-8')).hexdigest()
        train_hashes.add(h)

    clean_eval = []
    dupes = 0
    for entry in eval_set:
        h = hashlib.md5(entry.get("input", "").encode('utf-8')).hexdigest()
        if h in train_hashes:
            dupes += 1
        else:
            clean_eval.append(entry)
    print(f"  Dedup: removed {dupes:,} eval samples that overlapped with train")
    return train, clean_eval


# ──────────────────────────────────────────────────────────
# READERS
# ──────────────────────────────────────────────────────────
def read_hlstrans(path: str):
    """
    Reads the HLStrans.jsonl file.  This file stores concatenated JSON
    objects separated by literal '\\n' (backslash-n), NOT real newlines.
    We use json.JSONDecoder.raw_decode to stream through it.
    """
    print(f"  Reading HLStrans from {path} ...")
    decoder = json.JSONDecoder()
    records = []

    with open(path, 'r', encoding='utf-8') as f:
        buf = ""
        chunk_size = 64 * 1024 * 1024   # 64 MB chunks
        while True:
            chunk = f.read(chunk_size)
            if not chunk and not buf:
                break
            buf += chunk
            idx = 0
            while idx < len(buf):
                # Skip whitespace + literal \n separator
                while idx < len(buf) and buf[idx] in ' \t\r\n':
                    idx += 1
                if idx + 1 < len(buf) and buf[idx] == '\\' and buf[idx+1] == 'n':
                    idx += 2
                    continue
                if idx >= len(buf):
                    break
                try:
                    obj, end_idx = decoder.raw_decode(buf, idx)
                    records.append(obj)
                    idx = end_idx
                except json.JSONDecodeError:
                    # Need more data — keep remainder in buffer
                    break
            buf = buf[idx:]
            if not chunk:
                break

    print(f"    → Loaded {len(records):,} records")
    return records


def read_forgehls(path: str):
    """Read optional ForgeHLS JSON export (array of objects)."""
    if not os.path.isfile(path):
        print(f"  [skip] ForgeHLS file not found: {path}")
        return []
    print(f"  Reading ForgeHLS from {path} ...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"    → Loaded {len(data):,} records")
        return data
    print("    → Unexpected format, skipping")
    return []


def read_benchmarks(folder: str):
    """
    Read raw .c / .cpp benchmark files from CHStone, MachSuite, etc.
    Each file becomes one 'Logic' training sample.
    """
    if not os.path.isdir(folder):
        print(f"  [skip] Benchmarks folder not found: {folder}")
        return []
    print(f"  Scanning benchmarks in {folder} ...")
    samples = []
    for ext in ('*.c', '*.cpp', '*.h'):
        for filepath in glob.glob(os.path.join(folder, '**', ext), recursive=True):
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
            if len(code) < 50:
                continue
            samples.append({
                "instruction": (
                    "Analyze this C/C++ benchmark kernel and optimize it for "
                    "Xilinx Vivado HLS synthesis targeting an FPGA. Add appropriate "
                    "HLS pragmas for pipelining, unrolling, and array partitioning."
                ),
                "input": code,
                "output": ""   # Will need manual or model-assisted annotation
            })
    print(f"    → Found {len(samples):,} benchmark files")
    return samples


# ──────────────────────────────────────────────────────────
# MAIN BUILDER
# ──────────────────────────────────────────────────────────
def create_super_dataset():
    master_list = []

    # ── Pillar 1: Gold data (HLStrans) ────────────────────
    print("\n[Pillar 1] Merging Gold / Positive Samples …")
    hlstrans_records = read_hlstrans(INPUT_FILES["hlstrans"])
    for rec in hlstrans_records:
        master_list.append({
            "instruction": "Optimize this C code for Xilinx Versal HLS.",
            "input":  rec.get("input", ""),
            "output": rec.get("output", ""),
        })

    # ── Pillar 1b: ForgeHLS QoR data (capped) ────────────
    forge_records = read_forgehls(INPUT_FILES["forgehls"])
    if len(forge_records) > FORGE_QOR_CAP:
        print(f"  Capping ForgeHLS from {len(forge_records):,} → {FORGE_QOR_CAP:,}")
        forge_records = random.sample(forge_records, FORGE_QOR_CAP)
    for rec in forge_records:
        master_list.append({
            "instruction": rec.get("instruction",
                                   "Optimize this C code for Xilinx Versal HLS."),
            "input":  rec.get("input", rec.get("source", "")),
            "output": rec.get("output", rec.get("target", "")),
        })

    # ── Pillar 2: Logic / Benchmark data (skip empty outputs) ─
    print("\n[Pillar 2] Loading Complex Logic Benchmarks …")
    bench_samples = read_benchmarks(INPUT_FILES["benchmarks"])
    # Only include benchmarks that have an output (non-empty)
    bench_with_output = [b for b in bench_samples if b.get("output")]
    print(f"    → {len(bench_with_output):,} have output annotations")
    master_list.extend(bench_samples)  # keep all, empty-output filtered from debug pool

    # ── Pillar 3: Negative / Debugging data ───────────────
    print("\n[Pillar 3] Generating Debugging (Negative) Samples …")
    eligible = [e for e in master_list if e.get("output") and len(e["output"]) > 80]
    n_negative = min(len(eligible), int(len(eligible) * NEGATIVE_RATIO))
    debug_pool = random.sample(eligible, n_negative)

    debug_samples = []
    for entry in debug_pool:
        # Generate NEGATIVE_PASSES different corruptions per entry
        for _ in range(NEGATIVE_PASSES):
            broken_code = corrupt_hls_code(entry["output"])
            debug_samples.append({
                "instruction": random.choice(DEBUG_INSTRUCTIONS),
                "input":  broken_code,
                "output": entry["output"],
            })

    master_list.extend(debug_samples)
    print(f"  → Generated {len(debug_samples):,} debugging samples")

    # ── Also generate single-mode debug (easier examples) ─
    print("  Generating single-bug samples …")
    single_bug_pool = random.sample(eligible,
                                    min(len(eligible), int(len(eligible) * 0.25)))
    single_debug = []
    for entry in single_bug_pool:
        mode = random.choice(CORRUPTION_MODES)
        broken = corrupt_hls_code(entry["output"], mode=mode)
        single_debug.append({
            "instruction": random.choice(DEBUG_INSTRUCTIONS),
            "input":  broken,
            "output": entry["output"],
        })
    master_list.extend(single_debug)
    print(f"  → Generated {len(single_debug):,} single-bug samples")

    # ── Explain-the-bug samples (model learns to reason) ──
    print("  Generating explain-the-bug samples …")
    explain_pool = random.sample(eligible,
                                 min(len(eligible), int(len(eligible) * 0.15)))
    explain_samples = []
    for entry in explain_pool:
        modes = random.sample(CORRUPTION_MODES,
                              k=random.randint(1, 3))
        broken = entry["output"]
        for mode in modes:
            broken = corrupt_hls_code(broken, mode=mode)
        explanation_output = generate_explanation(modes, entry["output"])
        explain_samples.append({
            "instruction": random.choice(EXPLAIN_DEBUG_INSTRUCTIONS),
            "input":  broken,
            "output": explanation_output,
        })
    master_list.extend(explain_samples)
    print(f"  → Generated {len(explain_samples):,} explain-the-bug samples")

    # ── Pillar 4: Real bugfix pairs from git history ──────
    print("\n[Pillar 4] Mining real bugfix commits …")
    repo_dirs = glob.glob(os.path.join(BASE_DIR, "repos", "*"))
    git_samples = mine_git_bugfixes(repo_dirs)
    for s in git_samples:
        s.pop("_source", None)
    master_list.extend(git_samples)

    # ── Pillar 5: Known real-world HLS bug patterns ───────
    print("\n[Pillar 5] Adding known real-world HLS issue patterns …")
    known_samples = generate_known_issue_samples()
    master_list.extend(known_samples)

    total_debug = (len(debug_samples) + len(single_debug)
                   + len(explain_samples) + len(git_samples)
                   + len(known_samples))

    # ── Train / Eval split + deduplication ────────────────
    print(f"\nSplitting {len(master_list):,} total samples …")
    random.shuffle(master_list)
    n_eval = int(len(master_list) * EVAL_FRACTION)
    eval_set  = master_list[:n_eval]
    train_set = master_list[n_eval:]

    train_set, eval_set = dedup_by_input(train_set, eval_set)

    print(f"Writing {len(train_set):,} train → {OUTPUT_TRAIN}")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for entry in train_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Writing {len(eval_set):,} eval  → {OUTPUT_EVAL}")
    with open(OUTPUT_EVAL, 'w', encoding='utf-8') as f:
        for entry in eval_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n✓ Done!")
    print(f"  Train: {len(train_set):,} rows in {OUTPUT_TRAIN}")
    print(f"  Eval:  {len(eval_set):,}  rows in {OUTPUT_EVAL}")
    print(f"  Breakdown:")
    print(f"    Optimize (HLStrans):    {len(hlstrans_records):,}")
    print(f"    QoR (ForgeHLS capped):  {len(forge_records):,}")
    print(f"    Benchmarks:             {len(bench_samples):,}")
    print(f"    Debug (multi-bug):      {len(debug_samples):,}")
    print(f"    Debug (single-bug):     {len(single_debug):,}")
    print(f"    Debug (explain-bug):    {len(explain_samples):,}")
    print(f"    Debug (real git fixes): {len(git_samples):,}")
    print(f"    Debug (known issues):   {len(known_samples):,}")
    print(f"    TOTAL DEBUG:            {total_debug:,}  "
          f"({total_debug/len(master_list)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=1,
                        help='Number of corruption cycles (each uses a different seed)')
    args = parser.parse_args()

    if args.cycles <= 1:
        create_super_dataset()
    else:
        # Multi-cycle: run corruption generation multiple times with different
        # seeds, accumulate unique debug samples, write once at the end.
        all_train = []
        all_eval  = []
        seen_hashes = set()

        for cycle in range(args.cycles):
            print(f"\n{'='*60}")
            print(f"  CYCLE {cycle+1} / {args.cycles}  (seed={RANDOM_SEED + cycle})")
            print(f"{'='*60}")
            random.seed(RANDOM_SEED + cycle)
            create_super_dataset()

            # Read the just-written files and merge
            with open(OUTPUT_TRAIN, 'r', encoding='utf-8') as f:
                for line in f:
                    h = hashlib.md5(line.strip().encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_train.append(line)
            with open(OUTPUT_EVAL, 'r', encoding='utf-8') as f:
                for line in f:
                    h = hashlib.md5(line.strip().encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_eval.append(line)

        # Write merged results
        random.shuffle(all_train)
        random.shuffle(all_eval)
        with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
            f.writelines(all_train)
        with open(OUTPUT_EVAL, 'w', encoding='utf-8') as f:
            f.writelines(all_eval)

        print(f"\n{'='*60}")
        print(f"  MULTI-CYCLE MERGED RESULTS ({args.cycles} cycles)")
        print(f"{'='*60}")
        print(f"  Train: {len(all_train):,} unique rows")
        print(f"  Eval:  {len(all_eval):,} unique rows")
        print(f"  Total: {len(all_train)+len(all_eval):,}")
