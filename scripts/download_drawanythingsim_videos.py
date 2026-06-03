#!/usr/bin/env python3
"""
Download evaluation videos from wandb and organize for the project website.

Usage:
    conda activate bpp-website
    python scripts/download_videos.py \\
        --bpp        entity/project/run_id \\
        --icrt       entity/project/run_id \\
        --goal-image entity/project/run_id

Output structure:
    media/results/drawanything_sim/unseen_rollout/{model}/{trial}.mp4
    media/results/drawanything_sim/unseen_rollout/bpp_attention/{trial}.mp4
"""

import argparse
import json
import netrc
import os
import re
import shutil
import urllib.parse
from pathlib import Path

import requests
import wandb
from tqdm import tqdm


def _wandb_api_key() -> str | None:
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        return auth[2] if auth else None
    except Exception:
        return os.environ.get("WANDB_API_KEY")


def download_wandb_file(run: wandb.apis.public.Run, wandb_path: str, dest: Path) -> None:
    """Download a run file with properly URL-encoded path (handles ? # & etc in filenames)."""
    encoded = urllib.parse.quote(wandb_path, safe="/")
    url = f"https://api.wandb.ai/files/{run.entity}/{run.project}/{run.id}/{encoded}"
    key = _wandb_api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download eval videos from wandb into the website folder structure."
    )
    p.add_argument("--bpp",        default=None, metavar="ENTITY/PROJECT/RUN_ID",
                   help="wandb run path for the BPP model")
    p.add_argument("--icrt",       default=None, metavar="ENTITY/PROJECT/RUN_ID",
                   help="wandb run path for the ICRT model")
    p.add_argument("--goal-image", default=None, metavar="ENTITY/PROJECT/RUN_ID",
                   dest="goal_image",
                   help="wandb run path for the Goal Image model")
    p.add_argument("--task",  default="drawanything_sim",
                   help="task folder name (default: drawanything_sim)")
    p.add_argument("--split", default="unseen_rollout",
                   help="evaluation split folder name (default: unseen_rollout)")
    return p.parse_args()


TASK  = ""  # set from args in main()
SPLIT = ""

REPO_ROOT   = Path(__file__).parent.parent
OUTPUT_BASE = Path()  # set from args in main()

# wandb summary key prefixes for videos and attention maps
VIDEO_KEY_PREFIX     = "eval/test/videos/"
ATTENTION_KEY_PREFIX = "eval/test/attention_map/"


# ── Name cleanup ──────────────────────────────────────────────────────────────

def clean_trial_name(wandb_key: str) -> str:
    """
    Turn a raw wandb key into a short clean trial name (display label).

    Examples:
        eval/test/videos/draw_long4_giraffe_lower_seed_10000  →  giraffe
        eval/test/videos/draw_w_lower_seed_10000              →  w
    """
    name = wandb_key.rsplit("/", 1)[-1]
    name = re.sub(r"_(lower|upper)_seed_\d+$", "", name)
    name = re.sub(r"^draw_", "", name)
    name = re.sub(r"^(long|short)\d*_", "", name)
    return name.strip("_")


def safe_filenames(labels: list[str]) -> dict[str, str]:
    """
    Return {label: safe_filename} handling case-insensitive filesystem conflicts.
    e.g. 'Q' → 'Q_upper', 'q' → 'q_lower' when both exist; otherwise label == filename.
    """
    from collections import Counter
    colliding = {lw for lw, n in Counter(l.lower() for l in labels).items() if n > 1}
    result: dict[str, str] = {}
    for label in labels:
        if label.lower() in colliding:
            if label.isupper():
                result[label] = label + "_upper"
            elif label.islower():
                result[label] = label + "_lower"
            else:
                result[label] = label  # mixed-case, no standard suffix
        else:
            result[label] = label
    return result


# ── wandb helpers ─────────────────────────────────────────────────────────────

VIDEO_TYPES = {"video-file", "videos/separated"}


def find_video_entries(source: dict, prefix: str) -> dict[str, dict]:
    """Return {key: meta} for all video-type entries whose key starts with prefix."""
    return {
        k: v
        for k, v in source.items()
        if k.startswith(prefix)
        and isinstance(v, dict)
        and v.get("_type") in VIDEO_TYPES
    }


def list_files_for_prefix(run: wandb.apis.public.Run, prefix: str) -> dict[str, dict]:
    """
    Find video files via run.files() metadata listing — much faster than scan_history.
    Returns {key: {"_type": "video-file", "path": file_path}} using the highest-step file per key.
    """
    FILE_RE = re.compile(r"^media/videos/(.+)_(\d+)_[0-9a-f]+\.mp4$")
    latest: dict[str, tuple[int, str]] = {}
    for f in tqdm(run.files(), desc="  listing files", unit="file", ncols=80):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        key, step = m.group(1), int(m.group(2))
        if key.startswith(prefix):
            if key not in latest or step > latest[key][0]:
                latest[key] = (step, f.name)
    return {key: {"_type": "video-file", "path": path} for key, (_, path) in latest.items()}


def download_entries(
    run: wandb.apis.public.Run,
    entries: dict[str, dict],
    out_dir: Path,
    desc: str,
) -> list[dict[str, str]]:
    """Download video entries to out_dir. Returns [{label, file}] manifest sorted by label."""
    out_dir.mkdir(parents=True, exist_ok=True)

    labels    = [clean_trial_name(k) for k in entries]
    file_map  = safe_filenames(labels)
    manifest: list[dict[str, str]] = []

    for key, meta in tqdm(entries.items(), desc=f"  {desc}", unit="file", ncols=80):
        label = clean_trial_name(key)
        safe  = file_map[label]
        download_wandb_file(run, meta["path"], out_dir / f"{safe}.mp4")
        manifest.append({"label": label, "file": safe})

    return sorted(manifest, key=lambda x: x["label"])


# ── Per-run download ──────────────────────────────────────────────────────────

def process_run(
    run_path: str,
    model_id: str,
    api: wandb.Api,
) -> list[dict[str, str]]:
    print(f"\n{'─' * 64}")
    print(f"  model  : {model_id}")
    print(f"  run    : {run_path}")

    run     = api.run(run_path)
    summary = dict(run.summary)

    # ── regular videos ────────────────────────────────────────────
    out_dir       = OUTPUT_BASE / model_id
    video_entries = find_video_entries(summary, VIDEO_KEY_PREFIX)

    if not video_entries:
        print(f"  [!] Not in summary — listing run files ...")
        video_entries = list_files_for_prefix(run, VIDEO_KEY_PREFIX)

    if not video_entries:
        print(f"  [!] No video entries found for prefix '{VIDEO_KEY_PREFIX}'.")
        return []

    print(f"  videos : {len(video_entries)} found → {out_dir.relative_to(REPO_ROOT)}")
    written = download_entries(run, video_entries, out_dir, f"{model_id} videos")

    # ── attention maps (BPP only) ──────────────────────────────────
    if model_id == "bpp":
        attn_dir     = OUTPUT_BASE / "bpp_attention"
        attn_entries = find_video_entries(summary, ATTENTION_KEY_PREFIX)

        if not attn_entries:
            attn_entries = list_files_for_prefix(run, ATTENTION_KEY_PREFIX)

        if not attn_entries:
            print(f"  [!] No attention map videos found.")
        else:
            print(f"  attn   : {len(attn_entries)} found → {attn_dir.relative_to(REPO_ROOT)}")
            download_entries(run, attn_entries, attn_dir, "bpp attention")

    return written


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global OUTPUT_BASE, TASK, SPLIT

    args = parse_args()
    TASK  = args.task
    SPLIT = args.split
    OUTPUT_BASE = REPO_ROOT / "media" / "results" / TASK / SPLIT

    runs: dict[str, str] = {
        k: v for k, v in {
            "bpp":        args.bpp,
            "icrt":       args.icrt,
            "goal_image": args.goal_image,
        }.items() if v is not None
    }

    print("=" * 64)
    print("  BPP Website — wandb video downloader")
    print("=" * 64)

    # Clear only the model dirs being downloaded so partial re-runs don't wipe other models
    for model_id in runs:
        for d in [OUTPUT_BASE / model_id, OUTPUT_BASE / "bpp_attention"] if model_id == "bpp" else [OUTPUT_BASE / model_id]:
            if d.exists():
                print(f"Clearing {d.relative_to(REPO_ROOT)} ...")
                shutil.rmtree(d)

    api = wandb.Api()
    all_trials: list[dict[str, str]] = []

    for model_id, run_path in runs.items():
        trials = process_run(run_path, model_id, api)
        if trials and not all_trials:
            all_trials = trials  # use first model's trials as canonical manifest

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  Done! Files written:")
    for p in sorted(OUTPUT_BASE.rglob("*.mp4")):
        print(f"    {p.relative_to(REPO_ROOT)}")

    if all_trials:
        manifest = OUTPUT_BASE / "trials.json"
        manifest.write_text(json.dumps(all_trials, indent=2))
        print(f"\n  Trial manifest written to: {manifest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
