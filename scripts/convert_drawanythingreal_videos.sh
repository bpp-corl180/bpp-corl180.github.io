#!/bin/bash
# Convert HLG HDR iPhone MOV files to SDR H264 MP4.
#
# Usage:
#   bash scripts/convert_videos.sh <input_file> <output_file> [speed] [target_mb]
#
# Arguments:
#   input_file   Path to source .MOV/.mov file
#   output_file  Path to output .mp4 file
#   speed        Playback speed multiplier (default: 3)
#   target_mb    Target file size in MB (default: 5)
#
# Example:
#   bash scripts/convert_videos.sh input.MOV output.mp4 3 5

set -e

INPUT="$1"
OUTPUT="$2"
SPEED="${3:-3}"
TARGET_MB="${4:-5}"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: $0 <input_file> <output_file> [speed] [target_mb]"
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

echo "Input  : $INPUT"
echo "Output : $OUTPUT"
echo "Speed  : ${SPEED}x"
echo "Duration: ${DUR}s → $(python3 -c "print(round(float('$DUR')/$SPEED,1))")s"
echo "Bitrate: ${KBPS}kbps (target ${TARGET_MB}MB)"

ffmpeg -y -hwaccel none -i "$INPUT" \
  -vf "setpts=PTS/${SPEED},zscale=t=linear:npl=1000,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=reinhard:peak=1000:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,eq=gamma=1.3" \
  -r 30 \
  -c:v libx264 -preset slow -b:v "${KBPS}k" -maxrate "${MAXRATE}k" -bufsize "${BUFSIZE}k" \
  -color_trc bt709 -colorspace bt709 -color_primaries bt709 \
  -an "$OUTPUT"

echo "Saved: $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
