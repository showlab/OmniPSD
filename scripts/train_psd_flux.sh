#!/bin/bash
# Train text-to-PSD generation (Flux-Dev + RGBA-VAE LoRA).
# Run from the OmniPSD repository root:
#   cd /PATH/TO/OmniPSD
#   bash scripts/train_psd_flux.sh

export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes 4 --num_machines 1 \
  train/train.py \
  --dataset_base_path /PATH/TO/DATASET/psd_merge4 \
  --dataset_metadata_path /PATH/TO/DATASET/psd_merge4/flux_train_metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths '[
    "/PATH/TO/FLUX1-DEV/flux1-dev.safetensors",
    "/PATH/TO/FLUX1-DEV/text_encoder/model.safetensors",
    "/PATH/TO/FLUX1-DEV/text_encoder_2/",
    "/PATH/TO/RGBA_VAE/diffusion_pytorch_model.safetensors"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/PATH/TO/OUTPUT/psd_flux" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 128 \
  --align_to_opensource_format \
  --use_gradient_checkpointing \
  --save_steps 5000
