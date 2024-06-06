import argparse
import logging
import os
import pprint

import torch
import torch.onnx
import numpy as np
from torch import nn
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from datasets.palettes import get_palette

from third_party.unimatch.dataset.semi import SemiDataset
from model.builder import build_model, build_model2
from third_party.unimatch.supervised import predict
from datasets.classes import CLASSES
from third_party.unimatch.util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--ema', action='store_true')
parser.add_argument('--pred-path', default=None, type=str)
parser.add_argument('--logit-path', default=None, type=str)

def evaluate(model, loader, mode, cfg, pred_path=None, logit_path=None):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    palette = get_palette(cfg['dataset'])

    with torch.no_grad():
        for img, mask, id in tqdm(loader, total=len(loader)):
            file_name, lbl_name = id[0].split(' ')
            img = img  # 이미 CPU에서 실행되므로 cuda 호출 제거

            # Save input image
            if pred_path is not None:
                input_img_path = os.path.join(pred_path, 'input_images', lbl_name.split('/')[-1])
                os.makedirs(os.path.dirname(input_img_path), exist_ok=True)
                input_img = img[0].numpy().transpose(1, 2, 0).astype(np.uint8)
                Image.fromarray(input_img).save(input_img_path)

            pred, final = predict(model, img, mask, mode, cfg, return_logits=True)

            if logit_path is not None:
                logit_file = os.path.join(logit_path, lbl_name.split('/')[-1])\
                    .replace('.png', '.pt')
                os.makedirs(os.path.dirname(logit_file), exist_ok=True)
                torch.save(final.detach(), logit_file)  # cpu() 제거

            if pred_path is not None:
                pred_file = os.path.join(pred_path, lbl_name.split('/')[-1])
                os.makedirs(os.path.dirname(pred_file), exist_ok=True)
                np_pred = pred[0].numpy().astype(np.uint8)  # cpu() 제거
                output = Image.fromarray(np_pred).convert('P')
                output.putpalette(palette)
                output.save(pred_file)

            intersection, union, target = \
                intersectionAndUnion(pred.numpy(), mask.numpy(), cfg['nclass'], 255)  # cpu() 제거

            intersection_meter.update(intersection)
            union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class

def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = {**cfg, **vars(args), 'ngpus': 1}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    model = build_model2(cfg)
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))


    model = model  # 모델을 그대로 사용

    valset = SemiDataset(cfg, 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    if args.save_path != 'none':
        checkpoint = torch.load(os.path.join(args.save_path), map_location='cpu')  # GPU에서 저장된 체크포인트를 CPU로 로드
        if args.ema:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['ema_model'].items()}
        else:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        for k in list(checkpoint['model'].keys()):
            if 'clip_encoder' in k:
                del checkpoint['model'][k]
        model.load_state_dict(checkpoint['model'])

    #mIoU, iou_class = evaluate(model, valloader, 'original', cfg, pred_path=args.pred_path, logit_path=args.logit_path)

    #for (cls_idx, iou) in enumerate(iou_class):
        #logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                    #'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    #logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format('original', mIoU))

if __name__ == '__main__':
    main()
