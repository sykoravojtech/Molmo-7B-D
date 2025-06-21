# Molmo-7B-D

## Hardware requirements
has been tested to run on 24GB VRAM

### interactive node
```
qsub -I -l select=1:ncpus=1:ngpus=1:gpu_mem=24gb -l walltime=6:00:00
```

### run already created env
```
export TMPDIR=$SCRATCHDIR
export POETRY_HOME=$SCRATCHDIR/poetry
export POETRY_CACHE_DIR=$SCRATCHDIR/poetry/cache
export POETRY_VIRTUALENVS_PATH=$SCRATCHDIR/poetry/venvs
export PATH=$POETRY_HOME/bin:$PATH

curl -sSL https://install.python-poetry.org | python3 -

cd /storage/brno2/home/nademvit/vthesis/Molmo-7B-D

module add python/3.11.11-gcc-10.2.1-555dlyc

poetry env use $(which python)
poetry install

```

### run programs
```
poetry run python -m molmo_7b_d.test
```

### create env
https://www.youtube.com/watch?v=i8aD0Mprecw
```
export TMPDIR=$SCRATCHDIR
export POETRY_HOME=$SCRATCHDIR/poetry
export POETRY_CACHE_DIR=$SCRATCHDIR/poetry/cache
export POETRY_VIRTUALENVS_PATH=$SCRATCHDIR/poetry/venvs
export PATH=$POETRY_HOME/bin:$PATH

curl -sSL https://install.python-poetry.org | python3 -

cd /storage/brno2/home/nademvit/vthesis/Molmo-7B-D

module add python/3.11.11-gcc-10.2.1-555dlyc

poetry config virtualenvs.in-project true
poetry self add poetry-dotenv-plugin

poetry env use $(which python)
poetry install

poetry add torch transformers accelerate huggingface_hub pillow torchvision sentencepiece bitsandbytes fairscale fire blobfile einops

poetry add tensorflow
```