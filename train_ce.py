""""Python Script to train a cross entropy
model and save accordingly, to be used for
inference from the inference_ce.py Script"""

from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter, plot_results
from util import adjust_learning_rate, warmup_learning_rate, accuracy, confusion_matrix
from util import set_optimizer, save_model_strip
from networks.resnet_big import SupCEResNet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--save_best', action='store_true', help='saving best model')
    parser.add_argument('--save_last', action='store_true', help='saving last model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=600,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='300,400,500',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--n_cls', type=int, default=None)
    parser.add_argument('--mean', type=str, default='(0.5,0.5,0.5)',
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='(0.5,0.5,0.5)',
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    # data augmentation
    parser.add_argument('--crop_size', type=str, default='(240,320)', help='parameter for RandomResizedCrop')
    parser.add_argument('--crop_scale', type=str, default='(0.95, 1.0)',
                        help='crop scale for RandomResizedCrop in form of str tuple')
    parser.add_argument('--crop_ratio', type=str, default='(0.95, 1.05)',
                        help='crop ratio for RandomResizedCrop in form of str tuple')
    parser.add_argument('--degrees', type=int, default=15,
                        help='limit for degrees used in random rotation augmentation')
    parser.add_argument('--p_jitter', type=float, default=0.5, help='probability for colour jitter augmentation')
    parser.add_argument('--jitter', type=str, default='[0.4, 0.4, 0.4, 0.1]',
                        help='parameters for colour jitter augmentation as list [brightness, contrast, saturation, hue]')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    assert opt.data_folder is not None  # need to specify dataset path
    assert opt.n_cls is not None  # need to specify number of classes
    assert opt.model_name is not None  # need to specify model name

    # set the path according to the environment
    opt.trial = 0
    opt.experiment_folder = './runs/{}_{}/'.format(opt.model_name, opt.trial)

    # incrementally increase trial number
    folder_exists = True
    while folder_exists:
        if os.path.exists(opt.experiment_folder):
            opt.trial = int(opt.experiment_folder.split('_')[-1][:-1]) + 1
            opt.experiment_folder = './runs/{}_{}/'.format(opt.model_name, opt.trial)
        else:
            folder_exists = False

    opt.model_path = os.path.join(opt.experiment_folder, 'model')
    opt.metrics_path = os.path.join(opt.experiment_folder, 'metrics')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = opt.model_path
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.metrics_folder = opt.metrics_path
    if not os.path.isdir(opt.metrics_folder):
        os.makedirs(opt.metrics_folder)

    return opt


def set_loader(opt):
    # construct data loader
    mean = eval(opt.mean)
    std = eval(opt.std)
    crop_scale = eval(opt.crop_scale)
    crop_ratio = eval(opt.crop_ratio)
    crop_size = eval(opt.crop_size)
    jitter_args = eval(opt.jitter)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=crop_size, scale=crop_scale, ratio=crop_ratio),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=opt.degrees),
        transforms.RandomApply(
            [transforms.RandomChoice(
                [transforms.ColorJitter(*jitter_args), ]
            )], p=opt.p_jitter),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=crop_size, scale=(0.99, 1), ratio=(0.99, 1)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(root=opt.data_folder + '/train/',
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(root=opt.data_folder + '/val_easy/',  # TODO: change to proper path
                                       transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader, val_transform


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, min(opt.n_cls, 5)))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
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
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, min(opt.n_cls, 5)))
            conf_mat = confusion_matrix(conf_mat, output, labels)
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    class_acc = (conf_mat.diag() / conf_mat.sum(1)).cpu().numpy()

    acc = top1.avg.cpu().numpy()

    results = open(opt.results_path, 'a+')
    results.write(str(acc * 1e-2) + ' ')
    for res in class_acc:
        results.write(str(res) + ' ')
    results.write('\n')
    results.close()

    return losses.avg, top1.avg, conf_mat, class_acc


def main():
    best_acc = 0
    opt = parse_option()
    opt.results_path = os.path.join(opt.metrics_folder, 'results.txt')
    opt.plot_path = os.path.join(opt.metrics_folder, 'cross_entropy.png')
    if os.path.exists(opt.results_path):
        os.remove(opt.results_path)

    # build data loader
    train_loader, val_loader, _ = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # save run parameters
    with open(os.path.join(opt.metrics_folder, 'opt.yaml'), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluation
        loss, val_acc, conf_mat, class_acc = validate(val_loader, model, criterion, opt)
        print('Validation Accuracy: {:.f3}, Train Accuracy {:.f3}'.format(val_acc, train_acc))
        print('Validation Class Accuracy {}'.format(class_acc))
        print('Validation confusion matrix {}'.format(conf_mat))

        if val_acc > best_acc:
            best_acc = val_acc
            if opt.save_best:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_best_strip.pth')
                save_model_strip(model, save_file)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}_strip.pth'.format(epoch=epoch))
            save_model_strip(model, save_file)

    # save the last model
    if opt.save_last:
        save_file = os.path.join(
            opt.save_folder, 'last_strip.pth')
        save_model_strip(model, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))

    plot_results(opt.results_path, opt.plot_path)


if __name__ == '__main__':
    main()
