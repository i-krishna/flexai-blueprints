# Accept license & push dataset as per https://github.com/flexaihq/blueprints/blob/main/experiments/qlora-ft-on-a-language-model/README.md#step-2-preparing-the-dataset 

RUN_UUID="$(whoami)-$(uuidgen | cut -d '-' -f 1)"
RUN_NAME="llama3-1-training-ddp-${RUN_UUID}"

flexai training run "${RUN_NAME}" --repository-url https://github.com/flexaihq/blueprints --requirements-path code/causal-language-modeling-qlora/requirements.txt \
  --dataset llama-tokenized-oag \
  --secret HF_TOKEN=secret_kd \
  --nodes 1 --accels 2 \
  -- code/causal-language-modeling-qlora/train.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_name timdettmers/openassistant-guanaco \
    --tokenized_dataset_load_dir /input/llama-tokenized-oag \
    --dataset_text_field text \
    --load_in_4bit \
    --use_peft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir /output-checkpoint \
    --log_level info
