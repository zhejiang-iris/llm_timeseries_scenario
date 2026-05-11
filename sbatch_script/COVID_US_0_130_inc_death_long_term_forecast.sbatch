#!/bin/bash
#SBATCH --job-name=COVID_US_0_130_LTF
#SBATCH --account=alrodri0
#SBATCH --partition=spgpu
#SBATCH --time=7:00:00
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhejiang@umich.edu
#SBATCH --output=sbatch_logs/COVID_US_0_130_LTF_%j.out
#SBATCH --error=sbatch_logs/COVID_US_0_130_LTF_%j.err

module purge
source /home/zhejiang/.local/miniconda3/etc/profile.d/conda.sh
conda activate scenario

echo "Job started on $(hostname) at $(date)"
echo "Running in conda env: $CONDA_DEFAULT_ENV"

cd /home/zhejiang/scenario/Time-LLM

python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/COVID/ \
    --data_path US_0-130_inc_death.csv \
    --model_id COVID_US_0_130_inc_death_96_24 \
    --model TimeLLM \
    --data CUSTOM \
    --features S \
    --target observation \
    --freq w \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des Exp \
    --itr 1 \
    --d_model 32 \
    --d_ff 128 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --llm_layers 64 \
    --train_epochs 30 \
    --model_comment TimeLLM-COVID-US-0-130-inc-death

echo "Job finished at $(date)"