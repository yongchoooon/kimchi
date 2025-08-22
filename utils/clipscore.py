import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import os
from tqdm import tqdm
from glob import glob
import pandas as pd

from transformers import CLIPProcessor, CLIPModel

def load_images(image_dir, image_idx):
    num_synthetic = 3
    images = []
    
    for i in range(num_synthetic):
        image_path = os.path.join(image_dir, f'aug-{image_idx}-{i}.png')
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    return images
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_ID = "Bingsu/clip-vit-large-patch14-ko"
CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)

def get_clip_text_score(image, class_name, class_names):
    a_photo_of_a_class_names = [name for name in class_names]

    inputs = CLIP_PROCESSOR(
        text = a_photo_of_a_class_names,
        images = image,
        return_tensors = 'pt',
        padding = True
    ).to(DEVICE)

    with torch.no_grad():
        similarity = CLIP_MODEL(**inputs)
        score = similarity.logits_per_image.softmax(dim=1)

    clip_score_list = score.tolist()[0]
    clip_score_list = list(map(lambda x: round(x, 5), clip_score_list))

    clip_text_score = clip_score_list[class_names.index(class_name)]

    return clip_text_score

image_dir = "outputs/korean-4-datasets/"

class_names = sorted(list(set([filename.split('_')[0] for filename in os.listdir(image_dir)])))

clipscores = []

sorted_image_dir = sorted(os.listdir(image_dir))
for img in tqdm(sorted_image_dir):
    img_path = image_dir + img

    image = Image.open(img_path).convert('RGB')
    class_name = img.split('_')[0]
    score = get_clip_text_score(image, class_name, class_names)

    clipscores.append(score)

print(clipscores)
avg_clipscore = round(sum(clipscores) / len(clipscores), 4)
print(avg_clipscore)