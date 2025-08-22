# KIMCHI: Korean dIffusion Model Adapting Culture and HIstory
> **KIMCHI: 한국의 문화와 역사를 반영한 한국어 이미지 생성 모델**, 서용득, 권용근, 이노아, 정규헌, 조성우, 장시온, 최성철, 2025, 한국CDE학회 동계학술대회 논문집

<div align="center">
    <table width="100%">
        <tr>
            <td width="40%" align="center">
                <img width="100%" alt="preprocessing" src="figs/preprocessing.jpg">
            </td>
            <td width="60%" align="center">
                <img width="100%" alt="Framework" src="figs/main.jpg">
            </td>
        </tr>
    </table>
</div>

# Setup
```
cd kimchi
pip install -r requirements.txt
```

# Run
## Dataset name
- 한국적 이미지(Korean Images): 'yonggeun/korean-image-caption-dataset'
- 한국 랜드마크 이미지(Korean Landmark Images): 'yonggeun/korean-landmark-image-caption-dataset'
- 한국 유적지 이미지(Korean Historical Site Images): 'yonggeun/korean-historical-building-image-caption-dataset'
- 한국 음식 이미지(Korean Food Images): 'letgoofthepizza/traditional-korea-food-captioning'

## Train
```bash
python train_text_to_image_all_lora.py  --pretrained_model_name_or_path='stable-diffusion-v1-5/stable-diffusion-v1-5' --dataset_name='letgoofthepizza/traditional-korea-food-captioning' --caption_column='text'  --resolution=512 --random_flip  --train_batch_size=16  --num_train_epochs=3 --checkpointing_steps=500  --learning_rate=1e-04 --lr_scheduler='constant' --lr_warmup_steps=0  --seed=42  --output_dir='korean-food-dataset-2'  --validation_prompt='배추김치' --report_to='wandb'
```
- Korean Landmark Images and Korean Historical Site Images use `caption_column='LLaMA_translation'`
- Korean Images and Korean Food Images use `caption_column='text'`

## Inference
```bash
python inference.py
```

# Inference Examples
<div align="center">
    <img width="80%" alt="Inference Examples" src="figs/fig.jpg">
</div>

# Awards
🏆 **2025 한국CDE학회 동계학술대회 우수발표상 수상**
- 본 연구는 2025년 한국CDE학회 동계학술대회에서 우수발표상을 수상하였습니다.

**관련 링크:**
[한국CDE학회 논문 페이지](https://www.cde.or.kr/html/?pmode=BBBS0033900001&page=1&smode=view&seq=3968&searchValue=&schLimit=10&schCategory=&searchTitle=searchAll#:~:text=%C2%A0KIMCHI%3A%20%ED%95%9C%EA%B5%AD%EC%9D%98%20%EB%AC%B8%ED%99%94%EC%99%80%20%EC%97%AD%EC%82%AC%EB%A5%BC%20%EB%B0%98%EC%98%81%ED%95%9C%20%ED%95%9C%EA%B5%AD%EC%96%B4%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%83%9D%EC%84%B1%20%EB%AA%A8%EB%8D%B8)