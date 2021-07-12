from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

import torch
import torch.backends.cudnn as cudnn
from networks.resnet_big import SupCEResNet
from train_ce import set_loader
from util import AverageMeter, accuracy, confusion_matrix

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for inference')

    # model
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to pre-trained run folder')
    parser.add_argument('--save_path', type=str, default=None,
                        help='path where to save output')

    opt = parser.parse_args()
    # check if settings are correct
    assert opt.ckpt is not None  # need pre trained model
    assert opt.save_path is not None  # need to specify where to save output

    # load additional options from checkpoint
    train_opt_path = os.path.join(opt.ckpt, 'metrics/opt.yaml')
    with open(train_opt_path) as f:
        train_opt = yaml.load(f, Loader=yaml.FullLoader)
    opt = {**opt, **train_opt}  # concat dicts
    return opt


def load_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    model.eval()
    return model


def validate(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()
    conf_mat = torch.zeros(opt.n_cls, opt.n_cls)

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)

            # update metric
            acc1, acc5 = accuracy(output, labels, topk=(1, min(opt.n_cls, 5)))
            conf_mat = confusion_matrix(conf_mat, output, labels)
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print('Confusion Matrix')
    print(conf_mat)
    print('Class Accuracy')
    class_acc = (conf_mat.diag() / conf_mat.sum(1)).cpu().numpy()
    print(class_acc)


def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model
    model = load_model(opt)

    # validate
    validate(val_loader, model, opt)


if __name__ == '__main__':
    main()
