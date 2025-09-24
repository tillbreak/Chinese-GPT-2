#!/usr/bin/env bash
set -euo pipefail
# Full coverage heavier training run (Option C)
# Creates its own tokenized pieces (stride=512, more overlap) and trains 12 epochs.
# Usage (already automated via tmux command I launched):
#   tmux attach -t gpt_cov512   # to watch progress
# Logs stored under logs/

source .venv/bin/activate
cd "$(dirname "$0")"

LOG_DIR=logs
mkdir -p "$LOG_DIR"
RUN_NAME=full_cov512
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_${STAMP}.log"

echo "[INFO] Logging to $LOG_FILE"
echo "[INFO] Start time: $(date)" | tee -a "$LOG_FILE"

python train.py \
  --device 0,1 \
  --model_config config/model_config_small.json \
  --tokenizer_path cache/vocab_small.txt \
  --raw_data_path data/train.json \
  --tokenized_data_path data/tokenized_cov512/ \
  --raw \
  --epochs 12 \
  --batch_size 12 \
  --gradient_accumulation 2 \
  --stride 512 \
  --num_pieces 32 \
  --min_length 128 \
  --lr 8e-5 \
  --warmup_steps 3000 \
  --log_step 48 \
  --output_dir model_full_cov512/ \
  --writer_dir runs/full_cov512/ 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[INFO] Finished at: $(date) with exit code $EXIT_CODE" | tee -a "$LOG_FILE"

# If you want to continue training later with resume (example):
# python train.py --device 0,1 --resume_dir model_full_cov512/model_epoch6 --epochs 12 ... (adjust)
