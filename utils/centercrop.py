import os
import cv2


def center_crop(img, set_size=300):
    h, w, c = img.shape
    
    # 만약 이미지가 set_size보다 작으면 패딩 추가
    if h < set_size or w < set_size:
        pad_h = max(set_size - h, 0)
        pad_w = max(set_size - w, 0)
        
        # 이미지를 중앙에 배치하고 나머지 부분은 검은색(0)으로 채움
        img = cv2.copyMakeBorder(img, pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        h, w, c = img.shape  # 패딩 후 새로운 크기

    # set_size에 맞게 중앙 크롭
    mid_x, mid_y = w // 2, h // 2
    offset_x, offset_y = set_size // 2, set_size // 2

    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

src_img_dir = '/home/work/kimchi/fid/real_world_dataset'
save_img_dir = '/home/work/kimchi/fid/real_world_dataset_crop'
os.makedirs(save_img_dir, exist_ok=True)


for file in os.listdir(src_img_dir):
    try:
        src_img_path = os.path.join(src_img_dir, file)
        img = cv2.imread(src_img_path)
        
        crop_img = center_crop(img)
        save_image_path = os.path.join(save_img_dir, file)
        cv2.imwrite(save_image_path, crop_img)
    except Exception as e:
        print(f"{file}: {e}")
