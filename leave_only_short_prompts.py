import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import json

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

#################### configs ####################

output_dir = "outputs_yd/korean-4-datasets"
step = 37450

##################################################

# Prompts
prompts_hf_50000_path = f"{output_dir}/prompts_hf.json"

with open(prompts_hf_50000_path, 'r') as f:
    prompts_hf_50000 = json.load(f)

prompts_hf_50000 = {int(k): v for k, v in prompts_hf_50000.items()}

# Load CLIP
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_ID = "openai/clip-vit-base-patch16"

# 토크나이저 로드
tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)

def check_token_length(prompt, max_tokens=77):

    tokens = tokenizer.tokenize(prompt)
    token_count = len(tokens)
    return token_count > max_tokens, token_count

new_prompts_hf = {}
count_long_prompts = 0

progress_bar = tqdm(prompts_hf_50000.items(), desc="Checking token length")

# 77개 이상의 토큰을 가진 prompt만 남기기
for idx, prompt in prompts_hf_50000.items():
    is_long, token_count = check_token_length(prompt)
    if is_long == False:
        new_prompts_hf[idx] = prompt
    else:
        count_long_prompts += 1
    progress_bar.update(1)
    progress_bar.set_postfix({"count_long_prompts": count_long_prompts})


new_prompts_hf_length = len(new_prompts_hf)

print(f"기존 prompt 개수: {len(prompts_hf_50000)}")
print(f"77개 이하 prompt 개수: {new_prompts_hf_length}")

new_prompts_hf_path = f"{output_dir}/prompts_hf_{new_prompts_hf_length}.json"

if not os.path.exists(new_prompts_hf_path):
    with open(new_prompts_hf_path, 'w') as f:
        json.dump(new_prompts_hf, f, ensure_ascii=False, indent=4)