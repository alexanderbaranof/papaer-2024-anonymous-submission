#!/bin/bash
#SBATCH --job-name=calculate_generation_quality
#SBATCH --error=calculate_generation_quality.err
#SBATCH --output=calculate_generation_quality.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

module purge
module load Python
module load CUDA/12.2
module load cmake/3.21.3
source deactivate
conda activate llama_env

srun python3 -u calculate_generation_quality.py