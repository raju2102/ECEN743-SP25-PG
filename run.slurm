#!/bin/bash

#SBATCH --job-name=RunPG
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=8G
#SBATCH --output=%j.log

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate rl_env

cd $SCRATCH/ECEN743-SP25-PG

echo "Starting Main!!!"
python PG.py
echo "Done!!!"