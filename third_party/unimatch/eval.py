import argparse
import logging
import os
import pprint

import torch
import numpy as np
from PIL import Image
from model.builder import build_model
import yaml
from tqdm import tqdm
from datasets.palettes import get_palette
from third_party.unimatch.supervised import predict
from datasets.classes import CLASSES
from third_party.unimatch.util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

parser = argparse.ArgumentParser(description='Inference on a Single Image for Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--image-path', type=str, required=True)

def load_image(image_path):
    """ 이미지를 로드하고 모델 입력을 위해 변환합니다. """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image, dtype=np.float32)
    image = image.transpose((2, 0, 1))  # 채널을 첫 번째 차원으로 이동
    image = torch.from_numpy(image).unsqueeze(0) / 255.0  # 정규화 및 배치 차원 추가
    return image

def evaluate(model, img, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    palette = get_palette(cfg['dataset'])
    
    with torch.no_grad():
        img = img.to('cpu')  # CPU 사용 설정

        pred, final = predict(model, img, None, 'original', cfg, return_logits=True)
        np_pred = pred[0].numpy().astype(np.uint8)
        output = Image.fromarray(np_pred).convert('P')
        output.putpalette(palette)
        output.show()

        # 결과 이미지 저장 (선택적)
        output.save('output_segmentation.png')

        # 여기서는 intersection과 union을 계산하지 않습니다. (단일 이미지 평가)

def main():
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    
    img = load_image(args.image_path)
    evaluate(model, img, cfg)

if __name__ == '__main__':
    main()
