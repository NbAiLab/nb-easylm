JAX_PLATFORM_NAME=cpu python3 -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='' \
    --model_size='3b' \
    --output_dir='./'