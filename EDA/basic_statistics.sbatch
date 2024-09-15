#!/bin/bash
#SBATCH --job-name=basic_statistics
#SBATCH --error=basic_statistics.err
#SBATCH --output=basic_statistics.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

module purge
module load Python
module load CUDA/12.2
module load cmake/3.21.3
source deactivate
conda activate llama_env

srun python3 -u basic_statistics.py