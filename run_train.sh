python main.py \
    --weighted_keywords \
    --enable_generation \
    --inference_type unixcoder_with_rl \
    --generator_model_path deepseek-ai/deepseek-coder-1.3b-base \
    --generator_batch_size_per_gpu 16 \
    --extractor_model_path "Qwen/Qwen2.5-1.5B-Instruct"\
    --extractor_batch_size_per_gpu 16 \
    --extractor_max_context_length 1024 \
    --extractor_max_generation_length 64 \
    --batch_size 8 \
    --lr 3e-5 \
    --output_dir result_train/deepseekcoder_1.3b_qwen2.5_1.5b \
    2>&1|tee output_train.log