from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
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
    parser.add_argument('--run', type=str, default=None,
                        help='path to pre-trained run folder')
    parser.add_argument('--ckpt', type=str, default='best',
                        help='model that should be used from checkpoint, e.g. "best", "last", "epoch_50"')
    parser.add_argument('--save_path', type=str, default=None,
                        help='path where to save output')

    opt = parser.parse_args()
    # check if settings are correct
    assert opt.ckpt is not None  # need pre trained model
    assert opt.save_path is not None  # need to specify where to save output

    # load additional options from checkpoint
    train_opt_path = os.path.join(opt.run, 'metrics/opt.yaml')
    with open(train_opt_path) as f:
        train_opt = yaml.load(f, Loader=yaml.FullLoader)
    # concat dicts and convert to Namespace
    opt = argparse.Namespace(**{**vars(opt), **train_opt})
    return opt


def load_model(opt):
    print('Loading Model')
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    ckpt_path = os.path.join(opt.run, 'model/ckpt_{}_strip.pth'.format(opt.ckpt))
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    model.load_state_dict(state_dict)

    model.eval()
    return model


def validate(val_loader, model, opt):
    print('Start Validation')
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


def load_image(path, val_transforms):
    image = Image.open(path)
    image = val_transforms(image)
    image = image.unsqueeze(0)
    if torch.cuda.is_available():
        image = image.float().cuda()
    return image


def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader, val_transforms = set_loader(opt)

    # build model
    model = load_model(opt)
    softmax = torch.nn.Softmax(dim=1)

    # validate
    # validate(val_loader, model, opt)

    # predict single image
    image = load_image('../images/val_easy/70/574.png', val_transforms)
    prediction = model(image)
    probabilities = softmax(prediction)
    print('Made prediction')
    print(probabilities)


if __name__ == '__main__':
    main()
