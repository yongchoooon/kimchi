import os   
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda"

# Load pre-trained stable diffusion model
pretrained_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)

# Load LoRA weights
dataset_name = "outputs/korean-4-datasets"
step=37450

lora_path = f"./{dataset_name}/checkpoint-{step}/pytorch_lora_weights.safetensors"  

# Load CLIP model used for training
clip_model_path = "Bingsu/clip-vit-large-patch14-ko"
text_encoder = CLIPTextModel.from_pretrained(clip_model_path)
text_encoder.to(device)
tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)

# Set CLIP model to pipeline
pipeline.text_encoder = text_encoder
pipeline.tokenizer = tokenizer

pipeline.load_lora_weights(lora_path)

# Move pipeline to CUDA device
pipeline = pipeline.to(device)

# Inference
prompt = "한국의 항구에는 흰색 글자가 새겨진 빨간 등대, 푸른 하늘, 푸른 언덕, 수역, 배 등 생동감 넘치는 풍경이 펼쳐진다."
with torch.autocast(device):
    image = pipeline(prompt, num_inference_steps=50).images[0]

# Save result image
save_image_path = os.path.join(dataset_name, f"{prompt}_{step}.jpg")
image.save(save_image_path, 'JPEG')
