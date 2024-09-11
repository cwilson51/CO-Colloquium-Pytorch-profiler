#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --mem=8000M
#SBATCH --account=def-account

# Load modules
module load StdEnv/2023 cuda/12.2 python/3.11 scipy-stack/2024a 

echo
echo "Copying ImageNet data..."
### Copy ImageNet data to $SLURM_TMPDIR/imagenet

# Activate virtual environment
echo
echo "Activating virtual environment..."
echo
virtualenv --no-download $SLURM_TMPDIR/venv && source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirments.txt

BATCH_SIZE=128
# Run training script
echo "====================================================="
echo "Running training..."
echo "====================================================="
python resnet50_train.py $BATCH_SIZE --nWorkers $SLURM_CPUS_PER_TASK --skip_first 10 --warmup 10 --active 10 --compile


# Unmount data
fusermount -u $SLURM_TMPDIR/imagenet
