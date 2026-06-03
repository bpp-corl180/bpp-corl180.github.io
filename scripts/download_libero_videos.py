#!/usr/bin/env python3
"""
Download LIBERO-Gen BPP attention map videos from wandb for the project website.

Usage:
    conda activate bpp-website
    python scripts/download_libero_videos.py \\
        --combination          entity/project/run_id \\
        --chain                entity/project/run_id \\
        --chain-no-second-step entity/project/run_id

Output structure:
    media/results/libero_combination/bpp_attention/{split}/{task}.mp4
    media/results/libero_chain/bpp_attention/{split}/{task}.mp4
    media/results/libero_chain_no_second_step/bpp_attention/{split}/{task}.mp4
"""

import argparse
import json
import netrc
import os
import re
import shutil
import urllib.parse
from collections import defaultdict
from pathlib import Path

import requests
import wandb
from tqdm import tqdm


# ── Experiment config ─────────────────────────────────────────────────────────

EXPERIMENTS: dict[str, dict] = {
    "combination": {
        "folder":        "libero_combination",
        "seen_splits":   ["libero_spatial_selected_combinations_inverse_view"],
        "unseen_splits": ["libero_spatial_selected_combinations_view"],
    },
    "chain": {
        "folder":        "libero_chain",
        "seen_splits":   [
            "libero_goal_chain_firststep_view",
            "libero_goal_chain_secondstep_view",
            "libero_goal_chain_selected_inverse_view",
        ],
        "unseen_splits": ["libero_goal_chain_selected_view"],
    },
    "chain_no_second_step": {
        "folder":        "libero_chain_no_second_step",
        "seen_splits":   [
            "libero_goal_chain_firststep_view",
            "libero_goal_chain_selected_inverse_view",
        ],
        "unseen_splits": ["libero_goal_chain_selected_view"],
    },
}

SPLIT_LABELS: dict[str, str] = {
    "libero_spatial_selected_combinations_inverse_view": "Seen Combinations",
    "libero_spatial_selected_combinations_view":         "Unseen Combinations",
    "libero_goal_chain_firststep_view":                  "First Step",
    "libero_goal_chain_secondstep_view":                 "Second Step",
    "libero_goal_chain_selected_inverse_view":           "Seen Chains",
    "libero_goal_chain_selected_view":                   "Unseen Chains",
}


SEEN_PREFIX   = "attention_map/train/"
UNSEEN_PREFIX = "unseen/attention_map/test/"
VIDEO_TYPES   = {"video-file", "videos/separated"}

REPO_ROOT = Path(__file__).parent.parent


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download LIBERO-Gen BPP attention map videos from wandb."
    )
    p.add_argument("--combination",          default=None, metavar="ENTITY/PROJECT/RUN_ID")
    p.add_argument("--chain",                default=None, metavar="ENTITY/PROJECT/RUN_ID")
    p.add_argument("--chain-no-second-step", default=None, metavar="ENTITY/PROJECT/RUN_ID",
                   dest="chain_no_second_step")
    return p.parse_args()


# ── Download helper ───────────────────────────────────────────────────────────

def _wandb_api_key() -> str | None:
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        return auth[2] if auth else None
    except Exception:
        return os.environ.get("WANDB_API_KEY")


def download_wandb_file(run: wandb.apis.public.Run, wandb_path: str, dest: Path) -> None:
    """Download a run file with properly URL-encoded path (handles special chars)."""
    encoded = urllib.parse.quote(wandb_path, safe="/")
    url = f"https://api.wandb.ai/files/{run.entity}/{run.project}/{run.id}/{encoded}"
    key = _wandb_api_key()
    resp = requests.get(url, headers={"Authorization": f"Bearer {key}"} if key else {}, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


# ── Key parsing ───────────────────────────────────────────────────────────────

def parse_attn_key(key: str) -> tuple[str, str] | None:
    """
    Return (split_name, task_file) from an attention map key, or None if unrecognised.

    Seen  : attention_map/train/{split}/{task}_{num}
    Unseen: unseen/attention_map/test/{split}/{task}_{num}
    """
    for prefix in [SEEN_PREFIX, UNSEEN_PREFIX]:
        if key.startswith(prefix):
            rest  = key[len(prefix):]
            parts = rest.split("/", 1)
            if len(parts) == 2:
                split_name = parts[0]
                task_file  = re.sub(r"_\d+$", "", parts[1])
                return split_name, task_file
    return None


def task_label(task_file: str) -> str:
    return task_file.replace("_", " ")


# ── Video discovery ───────────────────────────────────────────────────────────

def find_video_entries(source: dict, prefix: str) -> dict[str, dict]:
    return {
        k: v for k, v in source.items()
        if k.startswith(prefix)
        and isinstance(v, dict)
        and v.get("_type") in VIDEO_TYPES
    }


def gather_attn_entries(run: wandb.apis.public.Run) -> dict[str, dict]:
    """
    Collect all attention map entries (seen + unseen).
    Checks summary first (instant), then falls back to listing run files (fast —
    just file metadata, not full history data).
    """
    summary = dict(run.summary)
    entries: dict[str, dict] = {}
    for prefix in [SEEN_PREFIX, UNSEEN_PREFIX]:
        found = find_video_entries(summary, prefix)
        if found:
            print(f"  {prefix}: {len(found)} entries in summary")
            entries.update(found)

    if entries:
        return entries

    # Fall back: list all run files and match by path pattern.
    # run.files() fetches only file metadata (name, size, url) — much faster than scan_history.
    print("  Not in summary — listing run files (faster than history scan) ...")
    FILE_RE = re.compile(r"^media/videos/(.+)_(\d+)_[0-9a-f]+\.mp4$")
    latest: dict[str, tuple[int, str]] = {}  # key → (step, file_path)

    for f in tqdm(run.files(), desc="  listing files", unit="file", ncols=80):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        key, step = m.group(1), int(m.group(2))
        for prefix in [SEEN_PREFIX, UNSEEN_PREFIX]:
            if key.startswith(prefix):
                if key not in latest or step > latest[key][0]:
                    latest[key] = (step, f.name)
                break

    if not latest:
        print("  [!] No attention map files found via file listing either.")
        return {}

    print(f"  Found {len(latest)} attention map files.")
    return {key: {"_type": "video-file", "path": path} for key, (_, path) in latest.items()}


# ── Per-experiment processing ─────────────────────────────────────────────────

def process_experiment(run_path: str, exp_key: str, api: wandb.Api) -> None:
    cfg        = EXPERIMENTS[exp_key]
    out_base   = REPO_ROOT / "media" / "results" / cfg["folder"] / "bpp_attention"
    all_splits = cfg["seen_splits"] + cfg["unseen_splits"]

    print(f"\n{'─' * 64}")
    print(f"  experiment : {exp_key}")
    print(f"  run        : {run_path}")
    print(f"  output     : {out_base.relative_to(REPO_ROOT)}")

    if out_base.exists():
        print(f"  Clearing existing output ...")
        shutil.rmtree(out_base)

    run     = api.run(run_path)
    entries = gather_attn_entries(run)

    if not entries:
        print("  [!] No attention map video entries found in this run.")
        return

    # Group by split name
    by_split: dict[str, list[tuple[str, str]]] = defaultdict(list)
    skipped = 0
    for key, meta in entries.items():
        parsed = parse_attn_key(key)
        if parsed is None or parsed[0] not in all_splits:
            skipped += 1
            continue
        split_name, task_file = parsed
        by_split[split_name].append((task_file, meta["path"]))

    if skipped:
        print(f"  ({skipped} entries skipped — not in expected splits)")

    # Download and build manifest
    manifest_splits = []
    for split_name in all_splits:
        tasks = sorted(set(by_split.get(split_name, [])))
        if not tasks:
            print(f"  [!] No videos for split '{split_name}'")
            continue

        label = SPLIT_LABELS.get(split_name, split_name)
        print(f"  {label} ({split_name}): {len(tasks)} tasks")

        task_entries: list[dict[str, str]] = []
        for task_file, wandb_path in tqdm(tasks, desc=f"    {label}", unit="file", ncols=80):
            dest = out_base / split_name / f"{task_file}.mp4"
            download_wandb_file(run, wandb_path, dest)
            task_entries.append({"label": task_label(task_file), "file": task_file})

        manifest_splits.append({
            "key":         split_name,
            "label":       label,
            "tasks": sorted(task_entries, key=lambda x: x["label"]),
        })

    manifest_path = REPO_ROOT / "media" / "results" / cfg["folder"] / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"splits": manifest_splits}, indent=2))
    print(f"  Manifest → {manifest_path.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    runs = {k: v for k, v in {
        "combination":          args.combination,
        "chain":                args.chain,
        "chain_no_second_step": args.chain_no_second_step,
    }.items() if v is not None}

    if not runs:
        print("No run IDs provided. Pass at least one of --combination, --chain, --chain-no-second-step.")
        return

    print("=" * 64)
    print("  BPP Website — LIBERO-Gen video downloader")
    print("=" * 64)

    api = wandb.Api()
    for exp_key, run_path in runs.items():
        process_experiment(run_path, exp_key, api)

    print(f"\n{'=' * 64}")
    print("  Done! Files written:")
    for exp_key in runs:
        base = REPO_ROOT / "media" / "results" / EXPERIMENTS[exp_key]["folder"]
        for p in sorted(base.rglob("*.mp4")):
            print(f"    {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
