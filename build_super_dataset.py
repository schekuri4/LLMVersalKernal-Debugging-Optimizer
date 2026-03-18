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
def mine_git_bugfixes(repo_paths: list) -> list:
    """
    Walk git history of cloned HLS repos, find commits that modified
    .c/.cpp files, and extract (before, after) pairs as real debugging data.
    """
    samples = []
    hls_keywords = ['pragma', 'hls', 'pipeline', 'unroll', 'partition',
                    'dataflow', 'interface', 'stream', 'ap_int', 'ap_uint']

    for repo_path in repo_paths:
        if not os.path.isdir(os.path.join(repo_path, '.git')):
            continue
        repo_name = os.path.basename(repo_path)
        print(f"  Mining {repo_name} ...")

        try:
            # Get commits that touched C/C++ files
            result = subprocess.run(
                ['git', 'log', '--all', '--pretty=format:%H', '--diff-filter=M',
                 '--', '*.c', '*.cpp', '*.h'],
                cwd=repo_path, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                continue
            commits = result.stdout.strip().split('\n')
            commits = [c for c in commits if c][:200]  # Cap at 200 commits
        except Exception:
            continue

        for commit_hash in commits:
            try:
                # Get list of changed files
                diff_result = subprocess.run(
                    ['git', 'diff', '--name-only', f'{commit_hash}~1', commit_hash,
                     '--', '*.c', '*.cpp', '*.h'],
                    cwd=repo_path, capture_output=True, text=True, timeout=15)
                if diff_result.returncode != 0:
                    continue
                changed_files = [f for f in diff_result.stdout.strip().split('\n') if f]
            except Exception:
                continue

            for cfile in changed_files[:5]:  # Max 5 files per commit
                try:
                    # Get before version
                    before = subprocess.run(
                        ['git', 'show', f'{commit_hash}~1:{cfile}'],
                        cwd=repo_path, capture_output=True, text=True, timeout=10)
                    # Get after version
                    after = subprocess.run(
                        ['git', 'show', f'{commit_hash}:{cfile}'],
                        cwd=repo_path, capture_output=True, text=True, timeout=10)

                    if before.returncode != 0 or after.returncode != 0:
                        continue

                    before_code = before.stdout
                    after_code = after.stdout

                    # Only keep if HLS-relevant
                    combined = (before_code + after_code).lower()
                    if not any(kw in combined for kw in hls_keywords):
                        continue
                    # Skip trivial changes (< 10 chars diff)
                    if abs(len(before_code) - len(after_code)) < 10 and \
                       before_code.strip() == after_code.strip():
                        continue
                    if len(before_code) < 50 or len(after_code) < 50:
                        continue

                    samples.append({
                        "instruction": random.choice(DEBUG_INSTRUCTIONS),
                        "input": before_code,
                        "output": after_code,
                        "_source": f"git:{repo_name}/{cfile}@{commit_hash[:8]}",
                    })
                except Exception:
                    continue

    print(f"    → Mined {len(samples):,} real bugfix pairs from git history")
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
    # Remove the _source field before adding to dataset
    for s in git_samples:
        s.pop("_source", None)
    master_list.extend(git_samples)

    total_debug = (len(debug_samples) + len(single_debug)
                   + len(explain_samples) + len(git_samples))

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
    print(f"    TOTAL DEBUG:            {total_debug:,}  "
          f"({total_debug/len(master_list)*100:.1f}%)")


if __name__ == "__main__":
    create_super_dataset()
