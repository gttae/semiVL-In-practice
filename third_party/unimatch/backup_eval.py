import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from datasets.palettes import get_palette

from third_party.unimatch.dataset.semi import SemiDataset
from model.builder import build_model
from third_party.unimatch.supervised import predict
from datasets.classes import CLASSES
from third_party.unimatch.util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--ema', action='store_true')
parser.add_argument('--pred-path', default=None, type=str)
parser.add_argument('--logit-path', default=None, type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def evaluate(model, loader, mode, cfg, pred_path=None, logit_path=None):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    palette = get_palette(cfg['dataset'])

    with torch.no_grad():
        for img, mask, id in tqdm(loader, total=len(loader)):
            file_name, lbl_name = id[0].split(' ')
            img = img.to('cpu')  # GPU 대신 CPU로 이동

            pred, final = predict(model, img, mask, mode, cfg, return_logits=True)

            if logit_path is not None:
                logit_file = os.path.join(logit_path, lbl_name.split('/')[-1])\
                    .replace('.png', '.pt')
                os.makedirs(os.path.dirname(logit_file), exist_ok=True)
                torch.save(final.detach().cpu(), logit_file)

            if pred_path is not None:
                pred_file = os.path.join(pred_path, lbl_name.split('/')[-1])
                os.makedirs(os.path.dirname(pred_file), exist_ok=True)
                np_pred = pred[0].cpu().numpy().astype(np.uint8)
                output = Image.fromarray(np_pred).convert('P')
                output.putpalette(palette)
                output.save(pred_file)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).to('cpu')
            reduced_union = torch.from_numpy(union).to('cpu')
            reduced_target = torch.from_numpy(target).to('cpu')

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # Legacy config support
    cfg.setdefault('text_embedding_variant', None)
    cfg.setdefault('pl_text', cfg['text_embedding_variant'])
    cfg['clip_encoder'] = None

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    if args.port is not None:
        raise NotImplementedError("Distributed training is not supported in this CPU example")
    
    rank, world_size = 0, 1
    local_rank = 0

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

    model = build_model(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to('cpu')  # GPU 대신 CPU로 이동

    valset = SemiDataset(cfg, 'val')

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=None)

    if args.save_path != 'none':
        checkpoint = torch.load(os.path.join(args.save_path), map_location=torch.device('cpu'))
        if args.ema:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['ema_model'].items()}
        else:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        for k in list(checkpoint['model'].keys()):
            if 'clip_encoder' in k:
                del checkpoint['model'][k]
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        if rank == 0:
            logger.info('************ WARNING: NO CHECKPOINT SPECIFIED')
    
    if 'eval_mode' in cfg:
        eval_mode = cfg['eval_mode']
    else:
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class = evaluate(
        model, valloader, eval_mode, cfg, 
        pred_path=args.pred_path,
        logit_path=args.logit_path)

    if rank == 0:
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))

if __name__ == '__main__':
    main()
