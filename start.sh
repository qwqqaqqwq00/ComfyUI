#!/bin/bash
# ComfyUI 启动脚本（Apple Silicon MPS 优化）
# 用法: ./start.sh

set -e

# 切换到脚本所在目录（ComfyUI 根目录）
cd "$(dirname "$0")"

# 激活 conda 环境
source /Users/wrd/miniforge3/etc/profile.d/conda.sh
conda activate comfyui

# 启动 ComfyUI
# 注: 不用 --cache-none --disable-smart-memory（在 MPS 上会导致频繁加载/卸载 + 内存不释放 → 系统卡死）
# 默认 RAM_PRESSURE 缓存模式会保留中间结果，减少重复加载
python main.py --auto-launch --listen --output-directory output_dir --enable-manager --reserve-vram 20 --fp16-unet
