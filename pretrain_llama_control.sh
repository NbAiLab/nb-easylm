#! /bin/bash

# Put your WANDB API key here to enable logging to wandb.
# export WANDB_API_KEY=''

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

BATCH_SIZE=256
SEQ_LENGTH=2048
TOTAL_TOKENS=65748775870
EPOCHS=2
STEPS=`python -c "print(round($EPOCHS * $TOTAL_TOKENS / ( $SEQ_LENGTH * $BATCH_SIZE )))"`
echo "Training for ($EPOCHS epochs * $TOTAL_TOKENS tokens / ($SEQ_LENGTH seq length * $BATCH_SIZE batch size)) = $STEPS steps"


python3 -m EasyLM.models.llama.llama_train \
    --jax_distributed.initialize_jax_distributed=True \
    --mesh_dim='1,16,1' \
    --dtype='bf16' \
    --total_steps="$STEPS" \
    --eval_freq=10000 \
    --log_freq=1000 \
    --save_model_freq=10000 \
    --save_milestone_freq=10000 \
    --load_llama_config='control' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.pretrained_model_name_or_path='mimir-project/tokenizer' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=1e-5 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --optimizer.adamw_optimizer.eps=1e-8 \
    --optimizer.adamw_optimizer.b1=0.9 \
    --optimizer.adamw_optimizer.b2=0.95 \
    --optimizer.adamw_optimizer.lr_warmup_steps=100 \
    --optimizer.adamw_optimizer.lr_decay_steps="$STEPS" \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_eos_token=True \
    --train_dataset.text_processor.add_bos_token=True \
    --train_dataset.huggingface_dataset.path='/share/mimir-base' \
    --train_dataset.huggingface_dataset.name='default' \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset.huggingface_dataset.seq_length="$SEQ_LENGTH" \
    --train_dataset.huggingface_dataset.batch_size="$BATCH_SIZE" \
    --eval_dataset.type='huggingface' \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.text_processor.add_eos_token=True \
    --eval_dataset.text_processor.add_bos_token=True \
    --eval_dataset.huggingface_dataset.path='/share/mimir-base' \
    --eval_dataset.huggingface_dataset.name='default' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --eval_dataset.huggingface_dataset.seq_length=2048 \
    --eval_dataset.huggingface_dataset.batch_size=128 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='Mímir EasyLM' \
    --logger.project="llama-control" \
    --logger.output_dir="gs://mimir-train-us/llama-control-checkpoint" \
    --logger.wandb_dir="$HOME/wandb" \
|& tee $HOME/output.txt
