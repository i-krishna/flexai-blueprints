# For testing debug-ssh interactive environment

# FlexAI training requires a requirements file to exist in the repository referenced by
# --repository-url. The TabICL repo only ships a pyproject.toml, so we use the FlexAI
# blueprints repo as the build context, and clone TabICL inside the job.
BLUEPRINTS_REPOSITORY_URL="https://github.com/flexaihq/blueprints"
BLUEPRINTS_REQUIREMENTS_PATH="code/causal-language-modeling/requirements.txt"

# TabICL repo containing the training code
# IMPORTANT: do NOT wrap the URL in angle brackets (<...>) in zsh/bash.
TABICL_REPOSITORY_URL="https://github.com/soda-inria/tabicl.git"

# Keep these aligned with your interactive debug-ssh command
WANDB_PROJECT="TabICL"
WANDB_NAME="Stage1"
WANDB_DIR="/output"
CHECKPOINT_DIR="/output"

# Clone repo & install dependencies 
git clone ${TABICL_REPOSITORY_URL} tabicl
cd tabicl
pip install -e .
cd src/tabicl

torchrun --standalone --nproc_per_node=1 train/run.py \
  --wandb_log False \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --wandb_dir "${WANDB_DIR}" \
  --wandb_mode offline \
  --device cuda \
  --dtype float32 \
  --np_seed 42 \
  --torch_seed 42 \
  --max_steps 100000 \
  --batch_size 512 \
  --micro_batch_size 4 \
  --lr 1e-4 \
  --scheduler cosine_warmup \
  --warmup_proportion 0.02 \
  --gradient_clipping 1.0 \
  --prior_type mix_scm \
  --prior_device cpu \
  --batch_size_per_gp 4 \
  --min_features 2 \
  --max_features 100 \
  --max_classes 10 \
  --max_seq_len 1024 \
  --min_train_size 0.1 \
  --max_train_size 0.9 \
  --embed_dim 128 \
  --col_num_blocks 3 \
  --save_perm_every 5000 \
  --checkpoint_dir "${CHECKPOINT_DIR}"
