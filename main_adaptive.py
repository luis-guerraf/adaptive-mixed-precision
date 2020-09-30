import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import models
from utils.model_profiling import model_profiling
from PIL import Image
import numpy as np
import torch.nn.utils as utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='tiny_imagenet', choices=['tiny_imagenet', 'imagenet'],
                    help='dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

alpha = 0.01
best_acc1 = 0
args = None

def main():
    global args
    args = parser.parse_args()

    # Set up
    set_up()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    num_classes = 1000 if args.dataset == 'imagenet' else 200
    print("=> creating model '{}'".format(args.arch))
    model = models.resnet_adaptive.__dict__[args.arch](num_classes=num_classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_RL = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)

    # Freeze main network
    model.apply(freeze_weights)
    model._modules['module'].precision_selector.apply(unfreeze_weights)

    # Only freezing the gradients will still update the running mean and var of the batch norms
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                              args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            # Resume from a vanilla 8bit_8bit network or from a checkpoint
            if '8bit.tar' in args.resume:
                # models.switchable_ops.switches = 2
                state_dict = models.remap_BN(checkpoint['state_dict'])
                model.load_state_dict(state_dict, strict=False)
                models.replicate_SBN_params(model)
            elif 'sandwich_a.tar' in args.resume:
                # models.switchable_ops.switches = 3
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'sandwich_w_a.tar' in args.resume:
                # models.switchable_ops.switches = 9
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch: {} acc1: {:0.2f})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'imagenet':
        datafolder = '/data/Datasets/IMAGENET/ImageNet_smallSize256/'
    elif args.dataset == 'tiny_imagenet':
        datafolder = '/data/Datasets/IMAGENET/tiny-imagenet-200/'

    traindir = os.path.join(datafolder, 'train')
    valdir = os.path.join(datafolder, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_crop = 224 if args.dataset == 'imagenet' else 64
    img_size = 256 if args.dataset == 'imagenet' else 64

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(img_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, criterion_RL, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            print(' * Acc@1 {:.3f}'.format(acc1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, criterion_RL, optimizer, epoch, args):
    batch_time = [AverageMeter('Time', ':6.3f'), AverageMeter('Time', ':6.3f')]
    losses = [AverageMeter('Loss', ':.2e'), AverageMeter('Loss', ':.2e')]
    top1 = [AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@1', ':6.2f')]
    top5 = [AverageMeter('Acc@5', ':6.2f'), AverageMeter('Acc@5', ':6.2f')]
    bitwidth_a = [AverageMeter('Bit_a', ':1.2f'), AverageMeter('Bit_a', ':1.2f')]
    bitwidth_w = [AverageMeter('Bit_w', ':1.2f'), AverageMeter('Bit_w', ':1.2f')]
    progress = [ProgressMeter(len(train_loader), [batch_time[0], losses[0], top1[0], top5[0], bitwidth_a[0], bitwidth_w[0]],
                                                    "Epoch: [{}]\t".format(epoch)),
                ProgressMeter(len(train_loader), [batch_time[1], losses[1], top1[1], top5[1], bitwidth_a[1], bitwidth_w[1]],
                              "Epoch: [{}]\t".format(epoch))]
    bns, _ = layers_list(model)

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        for switch in [8]:
            # Switch 4 is 8bit-8bit
            eightbit = 1 if switch == 4 else 0

            # Configure model
            list(map(lambda bn: setattr(bn, 'switch', switch), bns))
            end = time.time()

            output, precision, log_probs, entropies = model(images, eightbit)

            # loss = criterion(output, target)
            policy_loss = loss_RL(criterion_RL(output, target), precision, log_probs, entropies, eightbit)
            loss = policy_loss*0.01

            # PACT_L2_loss = list(map(pact_l2_loss, convs))
            # PACT_L2_loss = sum(PACT_L2_loss) / len(PACT_L2_loss)
            # loss += PACT_L2_loss*0.01

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses[eightbit].update(loss.item(), images.size(0))
            top1[eightbit].update(acc1[0], images.size(0))
            top5[eightbit].update(acc5[0], images.size(0))
            bitwidth_a[eightbit].update(precision[0].double().mean(), images.size(0))
            bitwidth_w[eightbit].update(precision[1].double().mean(), images.size(0))

            # measure elapsed time
            batch_time[eightbit].update(time.time() - end)

            # if epoch == 21:
            #     select_examples(images, precision)

    progress[0].display(i)
    progress[1].display(i)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = [AverageMeter('Time', ':6.3f'), AverageMeter('Time', ':6.3f')]
    losses = [AverageMeter('Loss', ':.2e'), AverageMeter('Loss', ':.2e')]
    top1 = [AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@1', ':6.2f')]
    top5 = [AverageMeter('Acc@5', ':6.2f'), AverageMeter('Acc@5', ':6.2f')]
    bitwidth_a = [AverageMeter('Bit_a', ':1.2f'), AverageMeter('Bit_a', ':1.2f')]
    bitwidth_w = [AverageMeter('Bit_w', ':1.2f'), AverageMeter('Bit_w', ':1.2f')]
    progress = [ProgressMeter(len(val_loader), [batch_time[0], losses[0], top1[0], top5[0], bitwidth_a[0], bitwidth_w[0]],
                                                    "Epoch: [{}]\t".format(epoch)),
                ProgressMeter(len(val_loader), [batch_time[1], losses[1], top1[1], top5[1], bitwidth_a[1], bitwidth_w[1]],
                              "Epoch: [{}]\t".format(epoch))]
    bns, _ = layers_list(model)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            for switch in [4, 8]:
                # Switch [4] is 8bit-8bit. Switch [8] is random-random
                eightbit = 1 if switch == 4 else 0

                # Configure model
                list(map(lambda bn: setattr(bn, 'switch', switch), bns))

                # compute output
                output, precision, _, _ = model(images, eightbit)

                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses[eightbit].update(loss.item(), images.size(0))
                top1[eightbit].update(acc1[0], images.size(0))
                top5[eightbit].update(acc5[0], images.size(0))
                bitwidth_a[eightbit].update(precision[0].double().mean(), images.size(0))
                bitwidth_w[eightbit].update(precision[1].double().mean(), images.size(0))

                # measure elapsed time
                batch_time[eightbit].update(time.time() - end)
                end = time.time()

        progress[0].display(i)
        progress[1].display(i)

    return top1[0].avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = 'pretrained/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'pretrained/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.arch == 'mobilenet_v2':
        for param_group in optimizer.param_groups:
            param_group['lr'] *= .98
    else:
        if args.dataset == 'imagenet':
            lr = args.lr * (0.1 ** (epoch // 15)) * (0.1 ** (epoch // 20))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Create list with conv2d
# NOTE: Make sure layers are declared in order in the model
def layers_list(model):
    BNs = list(filter(lambda x: isinstance(x, models.switchable_ops.SwitchableBatchNorm2d), [i for i in model.modules()]))
    convs = list(filter(lambda x: isinstance(x, nn.Conv2d), [i for i in model.modules()]))
    return BNs, convs


def pact_l2_loss(layer):
    if layer.__class__.__name__ == 'QuantizedConv2d':
        return layer.clip[0]
    else:
        return 0


control_variate = None
def loss_RL(batch_loss, precision, log_probs, entropies, eightbit):
    global control_variate
    if eightbit:
        return 0

    gamma = 1.0
    upper = 8
    num_layers = precision[0].size(1)
    cost_a = [1] * num_layers
    cost_w = [1] * num_layers

    policy_loss = []
    discounts = [gamma ** i for i in range(num_layers)]
    discounts.reverse()
    beta = alpha / num_layers

    control_variate = batch_loss.mean() if control_variate == None \
        else control_variate * 0.9 + batch_loss.mean() * 0.1
    temp = batch_loss.detach() - control_variate.detach()
    for layer in range(num_layers-1, 0, -1):
        # Reward from activations
        R_a = (upper - precision[0][:, layer]) * cost_a[layer]
        r_i = temp - (beta * R_a.float() * discounts[layer])
        policy_loss.append(log_probs[0][:, layer] * r_i - entropies[0][:, layer]*0.01)

        # Reward from weights
        R_w = (upper - precision[1][:, layer]) * cost_w[layer]
        r_i = temp - (beta * R_w.float() * discounts[layer])
        policy_loss.append(log_probs[1][:, layer] * r_i - entropies[1][:, layer]*0.01)

    policy_loss = torch.cat(policy_loss).mean()

    return policy_loss


def compute_num_operations(model):
    img_size = 256 if args.dataset == 'imagenet' else 64
    # model_profiling(
    #     model, img_size, img_size, verbose=getattr(FLAGS, 'model_profiling_verbose', True))
    pass


def save_img(img, file):
    img = (img.permute(1, 2, 0) - torch.min(img))
    img /= torch.max(img)
    img = img.cpu().detach().numpy()
    img *= 255
    img = Image.fromarray(np.uint8(img))
    img.save(file)


def select_examples(images, precision):
    very_simple_images = (precision[0].double().mean(dim=1) < 3).cpu().numpy() & \
                         (precision[1].double().mean(dim=1) < 3).cpu().numpy()
    simple_images = (precision[0].double().mean(dim=1) < 4).cpu().numpy() & \
                    (precision[1].double().mean(dim=1) < 4).cpu().numpy()
    complex_images = (precision[0].double().mean(dim=1) > 5).cpu().numpy() & \
                     (precision[1].double().mean(dim=1) > 5).cpu().numpy()
    very_complex_images = (precision[0].double().mean(dim=1) > 6).cpu().numpy() & \
                          (precision[1].double().mean(dim=1) > 6).cpu().numpy()

    for i, img in enumerate(very_simple_images):
        if img:
            file = 'paper_related/very_simple_images/' + str(i) + '.png'
            save_img(images[i], file)
    for i, img in enumerate(simple_images):
        if img:
            file = 'paper_related/simple_images/' + str(i) + '.png'
            save_img(images[i], file)
    for i, img in enumerate(complex_images):
        if img:
            file = 'paper_related/complex_images/' + str(i) + '.png'
            save_img(images[i], file)
    for i, img in enumerate(very_complex_images):
        if img:
            file = 'paper_related/very_complex_images/' + str(i) + '.png'
            save_img(images[i], file)


def set_up():
    models.switchable_ops.switches = 9
    pass


def freeze_weights(layer):
    for param in layer.parameters():
        param.requires_grad = False


def unfreeze_weights(layer):
    for param in layer.parameters():
        param.requires_grad = True


if __name__ == '__main__':
    main()
