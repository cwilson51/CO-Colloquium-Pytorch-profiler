# CO-Colloquium-Pytorch-profiler

## Job Scripts

The job scripts `imagenet_val.sh` and `imagenet_train.sh` will run jobs to create the traces seen in the presentation (traces are also included). The scripts create a virtual environment using the `requirements.txt` file in this repo.

You'll need to provide your account by modifying the header `#SBATCH --account=def-account`. If you are not on Narval, you will have to change the GPU request and modify the batch size accordingly

## Computing environment

If you'd like to recreate the computing environment on Alliance clusters follow the `module load` and `virtualenv` instructions below:

```bash
# Load modules
module load StdEnv/2023 cuda/12.2 python/3.11 scipy-stack/2024a 

# Create/Activate virtual environment
virtualenv --no-download ENV && source ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirments.txt
```



Also see the official documentation for [PyTorch profiler]('https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html') and [Holistic Trace Analysis](https://hta.readthedocs.io/en/latest/')