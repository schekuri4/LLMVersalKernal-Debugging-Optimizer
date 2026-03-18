"""
Super-Dataset Builder for HLS Training (3-Pillar Approach)
==========================================================
Pillars:
  1. Gold    – Best-case optimized HLS code (from HLStrans + ForgeHLS)
  2. Logic   – Complex benchmark kernels (CHStone, MachSuite, Vitis examples)
  3. Negative – Corrupted code the model learns to fix (auto-generated)

Input:
  - datasets/HLStrans.jsonl  (existing ~124k records, concatenated JSON)
  - data/forge_hls_exported.json (optional, ForgeHLS export)
  - data/raw_benchmarks/       (optional, .c/.cpp files from CHStone/MachSuite)

Output:
  - super_hls_train.jsonl  (standard newline-delimited JSONL)
"""

import json
import os
import re
import random
import glob
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

OUTPUT_FILE = os.path.join(BASE_DIR, "super_hls_train.jsonl")

# What fraction of positive samples to corrupt for debugging pillar
NEGATIVE_RATIO = 0.50          # 50 % of eligible → negative samples
NEGATIVE_PASSES = 3            # Generate N debug variants per sampled entry
RANDOM_SEED    = 42            # For reproducibility

random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────
# CORRUPTER  –  turns good HLS into broken "software" code
# ──────────────────────────────────────────────────────────
# Distinct corruption strategies — each produces a different class of bug
CORRUPTION_MODES = [
    "strip_pragmas",
    "revert_types",
    "inject_malloc",
    "inject_recursion",
    "swap_stream",
    "remove_pipeline",
    "add_pointer_cast",
    "variable_length_array",
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
]


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

    # ── Pillar 1b: Optional ForgeHLS ─────────────────────
    forge_records = read_forgehls(INPUT_FILES["forgehls"])
    for rec in forge_records:
        master_list.append({
            "instruction": rec.get("instruction",
                                   "Optimize this C code for Xilinx Versal HLS."),
            "input":  rec.get("input", rec.get("source", "")),
            "output": rec.get("output", rec.get("target", "")),
        })

    # ── Pillar 2: Logic / Benchmark data ──────────────────
    print("\n[Pillar 2] Loading Complex Logic Benchmarks …")
    bench_samples = read_benchmarks(INPUT_FILES["benchmarks"])
    master_list.extend(bench_samples)

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

    # ── Shuffle & Write ───────────────────────────────────
    print(f"\nShuffling {len(master_list):,} total samples …")
    random.shuffle(master_list)

    print(f"Writing to {OUTPUT_FILE} …")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in master_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n✓ Done!  Created {len(master_list):,} rows in {OUTPUT_FILE}")
    print(f"  Breakdown:")
    print(f"    Gold (HLStrans):     {len(hlstrans_records):,}")
    print(f"    Gold (ForgeHLS):     {len(forge_records):,}")
    print(f"    Logic (Benchmarks):  {len(bench_samples):,}")
    print(f"    Negative (Debug):    {len(debug_samples):,}")


if __name__ == "__main__":
    create_super_dataset()
