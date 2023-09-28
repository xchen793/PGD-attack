#!/bin/bash
#SBATCH --job-name=text-attack
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:3
#SBATCH --mem=32G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python ./EvaluateAttack.py --start_index 100 --num_samples 200 --dataset yelp
"
