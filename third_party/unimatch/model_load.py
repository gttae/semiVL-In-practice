import argparse
import torch
import yaml
from model.builder import build_model

parser = argparse.ArgumentParser(description='Load a pretrained model for Semantic Segmentation')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained model weights')

def main():
    args = parser.parse_args()
    
    # 설정 파일을 로드합니다.
    with open(args.config, "r") as file:
        cfg = yaml.load(file, Loader=yaml.Loader)
    
    # 설정에 기반한 모델을 구성합니다.
    model = build_model(cfg)
    
    # 사전 학습된 모델 가중치를 로드합니다.
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    # 모델을 평가 모드로 설정합니다.
    model.eval()
    
    # 로드된 모델의 정보를 출력 (선택 사항)
    print("Model loaded successfully with the following configuration:")
    print(cfg)

if __name__ == '__main__':
    main()
