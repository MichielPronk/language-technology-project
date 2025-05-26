#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

module purge
module load Python/3.11.5-GCCcore-13.2.0
cd $TMPDIR

cp -r /scratch/s3737101/ltp .
cd ltp

source /scratch/s3737101/venvs/ltp/bin/activate
export HF_TOKEN=$HF_TOKEN

python main.py --split test --model llama
cp predictions*.json $HOME/ltp
