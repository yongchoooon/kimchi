#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module

import itertools
import importlib.util

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "letgoofthepizza/traditional-korea-food-captioning": ("image", "text"),
    'yonggeun/korean-landmark-image-caption-dataset': ('image', 'text', 'korean_translation', 'LLaMA_generation', 'LLaMA_translation')
}

def main():

    parser = argparse.ArgumentParser("Few-Shot Baseline")
    parser.add_argument("--config", type=str, required=True, default=None)

    args = parser.parse_args()

    config_name = args.config
    config_file_path = os.path.join("configs", f"{config_name}.py")
    
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file {config_file_path} not found")

    module_name = os.path.splitext(os.path.basename(config_file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # Sanity checks
    if cfg.dataset_name is None and cfg.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    logging_dir = Path(cfg.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        # log_with=cfg.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler") # config : stable-diffusion-v1-5/stable-diffusion-v1-5
    
    text_encoder = CLIPTextModel.from_pretrained(
        "Bingsu/clip-vit-large-patch14-ko", 
        revision=cfg.revision, # config : None
        cache_dir='/home/work/kimchi/caches'
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "Bingsu/clip-vit-large-patch14-ko", 
        revision=cfg.revision,
        cache_dir='/home/work/kimchi/caches'
    )
    
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, # config : stable-diffusion-v1-5/stable-diffusion-v1-5
        subfolder="vae", 
        revision=cfg.revision, # config : None
        variant=cfg.variant, # config : None
        cache_dir='/home/work/kimchi/caches'
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=cfg.revision, 
        variant=cfg.variant,
        cache_dir='/home/work/kimchi/caches'
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    


    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    print("accelerator.mixed_precision : ", accelerator.mixed_precision)
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    for param in text_encoder.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=cfg.rank, # config : 4
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    text_encoder_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    text_encoder.add_adapter(text_encoder_lora_config)
        
    
    if cfg.mixed_precision == "fp16": # config : None
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)
        cast_training_params(text_encoder, dtyp=torch.float32)


    lora_layers_unet = filter(lambda p: p.requires_grad, unet.parameters())
    # Adding text_encoder's LoRA parameters to the optimizer
    lora_layers_text_encoder = filter(lambda p: p.requires_grad, text_encoder.parameters())
    

    # 두 모델의 파라미터를 결합
    all_lora_layers = itertools.chain(lora_layers_unet, lora_layers_text_encoder)

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        all_lora_layers,
        lr=cfg.learning_rate, # config : 1e-4
        betas=(cfg.adam_beta1, cfg.adam_beta2), # config : 0.9, 0.999
        weight_decay=cfg.adam_weight_decay, # config : 1e-2
        eps=cfg.adam_epsilon, # config : 1e-08
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if cfg.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.dataset_name, # config : 수정
            cfg.dataset_config_name, # config : None
            cache_dir='/home/work/kimchi/caches',
            data_dir=cfg.train_data_dir, # config : None
            split="train",
            streaming=False,  # 스트리밍 모드 비활성화
        )
        
    else:
        data_files = {}
        if cfg.train_data_dir is not None:
            data_files["train"] = os.path.join(cfg.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir='/home/work/kimchi/caches',
        )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    
    column_names = dataset.column_names # TODO : 데이터셋별로 class명과 캡션이 어떻게 되어있는지 확인해야 함. 

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(cfg.dataset_name, None)
    if cfg.image_column is None: # config : 'image'
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = cfg.image_column # config : 'image'
        if image_column not in column_names: # False
            raise ValueError(
                f"--image_column' value '{cfg.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if cfg.caption_column is None: # config : 'text'
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = cfg.caption_column # config : 'text'
        if caption_column not in column_names: # False
            raise ValueError(
                f"--caption_column' value '{cfg.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.resolution) if cfg.center_crop else transforms.RandomCrop(cfg.resolution), # configs : False
            transforms.RandomHorizontalFlip() if cfg.random_flip else transforms.Lambda(lambda x: x), # configs : True
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        
        return examples

    with accelerator.main_process_first():
        if cfg.max_train_samples is not None: # config : None
            dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_train_samples))
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train_batch_size, # configs : 16
        num_workers=cfg.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps) # configs : 4
    if cfg.max_train_steps is None: # configs : None
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch # configs : 3
        # Food : 3 * 2354 = 7062
        overrode_max_train_steps = True
        print("len(train_dataloader) : ", len(train_dataloader)) # Food : 9414
        print("cfg.gradient_accumulation_steps : ", cfg.gradient_accumulation_steps) # 4
        print("num_update_steps_per_epoch : ", num_update_steps_per_epoch) # Food : 2354
        print("cfg.num_train_epochs : ", cfg.num_train_epochs) # 3

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler, # configs : "constant"
        optimizer=optimizer, # AdamW
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes, # configs : 0
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch # cfg.max_train_steps가 configs에서 None일 경우 그대로 유지됨

    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps # 16 * 1 * 4 = 64

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}") # Food : 150,610 => (batch 16 * len 9414)
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}") # configs : 3
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}") # configs : 16
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}") # 64
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}") # configs : 4
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}") # Food : 7062
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint: # configs : None
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if cfg.noise_offset: # configs : 0
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) # noise_scheduler.config.num_train_timesteps : 1000
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if cfg.prediction_type is not None: # configs : None
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=cfg.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon": # epsilon
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if cfg.snr_gamma is None: # configs : None
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = all_lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0: # configs : 500
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None: # configs : None
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        
                        ## folder 만들어주기 
                        os.makedirs(save_path, exist_ok=True)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )
                        
                        unwrapped_text_encoder = unwrap_model(text_encoder)
                        text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_text_encoder)
                        )
                        
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            text_encoder_lora_layers=text_encoder_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        
        unwrapped_text_encoder = unwrap_model(text_encoder)
        text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_text_encoder)
)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=cfg.output_dir,
            text_encoder_lora_layers=text_encoder_lora_state_dict,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()