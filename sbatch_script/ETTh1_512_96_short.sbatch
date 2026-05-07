#!/bin/bash
#SBATCH --job-name=ETTh1_512_96_short
#SBATCH --account=alrodri0
#SBATCH --partition=gpu
#SBATCH --time=7:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhejiang@umich.edu
#SBATCH --output=log/ETTh1_512_96_short_%j.out
#SBATCH --error=log/ETTh1_512_96_short_%j.err


module purge
source /home/zhejiang/.local/miniconda3/etc/profile.d/conda.sh
conda activate scenario

echo "Job started on $(hostname) at $(date)"
echo "Running in conda env: $CONDA_DEFAULT_ENV"

cd /home/zhejiang/scenario/Time-LLM
python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_512_96 \
    --model TimeLLM \
    --data ETTh1 \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 96 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des Exp \
    --itr 1 \
    --d_model 32 \
    --d_ff 128 \
    --batch_size 1 \
    --learning_rate 0.01 \
    --llm_layers 1 \
    --train_epochs 1 \
    --model_comment TimeLLM-ETTh1

echo "Job finished at $(date)"