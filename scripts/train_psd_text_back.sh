#!/bin/bash
# Train text-layer background inpainting (Flux-Kontext LoRA).
# Run from the OmniPSD repository root:
#   cd /PATH/TO/OmniPSD
#   bash scripts/train_psd_text_back.sh

export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
  --num_processes 2 --num_machines 1 \
  train/train.py \
  --dataset_base_path /PATH/TO/DATASET/text \
  --dataset_metadata_path /PATH/TO/DATASET/text/back_metadata.csv \
  --data_file_keys "image,control" \
  --max_pixels 1048580 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-Kontext-dev:flux1-kontext-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/PATH/TO/OUTPUT/text_back" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 128 \
  --align_to_opensource_format \
  --extra_inputs "control" \
  --use_gradient_checkpointing \
  --save_steps 5000 \
  --lora_checkpoint "/PATH/TO/PRETRAINED_LORA/text_back.safetensors"
