#!/bin/bash
# Convert video files to SDR H264 MP4 with audio removed.
# Handles both HDR iPhone MOV files and already-SDR MP4 files.
#
# Usage:
#   bash scripts/convert_laundry_videos.sh <input_file> <output_file> [speed] [target_mb] [crop]
#
# Arguments:
#   input_file   Path to source video file
#   output_file  Path to output .mp4 file
#   speed        Playback speed multiplier (default: 1)
#   target_mb    Target file size in MB (default: 8)
#   crop         FFmpeg crop filter string, e.g. "crop=iw*0.8:ih*0.8:0:ih*0.1" (default: none)
#
# Examples:
#   bash scripts/convert_laundry_videos.sh input.mp4 output.mp4 2 8
#   bash scripts/convert_laundry_videos.sh input.MOV output.mp4 1 8 "crop=iw*0.8:ih*0.8:0:ih*0.1"

set -e

INPUT="$1"
OUTPUT="$2"
SPEED="${3:-1}"
TARGET_MB="${4:-8}"
CROP="${5:-}"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: $0 <input_file> <output_file> [speed] [target_mb] [crop]"
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Error: input file not found: $INPUT"
  exit 1
fi

DUR=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$INPUT" | tr -d ',\n')
KBPS=$(python3 -c "print(int($TARGET_MB * 8 * 1024 / (float('$DUR') / $SPEED)))")
MAXRATE=$((KBPS * 3 / 2))
BUFSIZE=$((KBPS * 3))

# Detect HDR by checking color_transfer (smpte2084 = PQ, arib-std-b67 = HLG)
COLOR_TRC=$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=color_transfer -of csv=p=0 "$INPUT" | tr -d ',\n')

# Build crop prefix for filter chain
if [ -n "$CROP" ]; then
  CROP_PREFIX="${CROP},"
else
  CROP_PREFIX=""
fi

echo "Input     : $INPUT"
echo "Output    : $OUTPUT"
echo "Speed     : ${SPEED}x"
echo "Duration  : ${DUR}s → $(python3 -c "print(round(float('$DUR')/$SPEED,1))")s"
echo "Bitrate   : ${KBPS}kbps (target ${TARGET_MB}MB)"
echo "Color TRC : ${COLOR_TRC}"
echo "Crop      : ${CROP:-none}"

if echo "$COLOR_TRC" | grep -qE "smpte2084|arib-std-b67|bt2020"; then
  echo "Mode      : HDR → SDR tonemap"
  VF="${CROP_PREFIX}setpts=PTS/${SPEED},zscale=t=linear:npl=1000,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=reinhard:peak=1000:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,eq=gamma=1.15"
else
  echo "Mode      : SDR passthrough (re-encode only)"
  VF="${CROP_PREFIX}setpts=PTS/${SPEED},format=yuv420p"
fi

ffmpeg -y -hwaccel none -i "$INPUT" \
  -vf "$VF" \
  -r 30 \
  -c:v libx264 -preset slow -b:v "${KBPS}k" -maxrate "${MAXRATE}k" -bufsize "${BUFSIZE}k" \
  -color_trc bt709 -colorspace bt709 -color_primaries bt709 \
  -an "$OUTPUT"

echo "Saved: $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
