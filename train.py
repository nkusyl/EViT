import time
import json
import torch
import argparse
import datetime
import numpy as np
from pathlib import Path
from timm.data import Mixup

import utils
from model.EViT import build_backbone
from engine import evaluate, train_one_epoch

from samplers import RASampler
from datasets import build_dataset
import torch.backends.cudnn as cudnn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma, ApexScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parsers = argparse.ArgumentParser('training and evaluation script', add_help=False)

    # important parameters
    parsers.add_argument('--save_path', default="/home/ubuntu/Datasets/EViT-main/save_path", help='path where to save, empty for no saving')
    parsers.add_argument('--batch_size', default=128, type=int)
    parsers.add_argument('--epochs', default=300, type=int)
    parsers.add_argument('--model', default=None, type=str, help='Name of model to train')
    parsers.add_argument('--input_size', default=224, type=int, help='images input size')
    parsers.add_argument('--dataset_name', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'], type=str, help='Image Net dataset path')
    parsers.add_argument('--datasets_path', default=None, type=str, help='dataset path')
    parsers.add_argument('--inat-category', default='name', choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'], type=str, help='semantic granularity')
    parsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    parsers.add_argument('--resume', default=None, help='resume from checkpoint')
    parsers.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parsers.add_argument('--eval', default=False, type=bool)
    parsers.add_argument('--num_workers', default=1, type=int)
    parsers.add_argument('--test_freq', type=int, default=30)
    parsers.add_argument('--test_epoch', type=int, default=260)
    parsers.add_argument('--apex_amp', action='store_true', default=False, help='Use NVIDIA Apex AMP mixed precision')

    # ema parameters
    parsers.add_argument('--model_ema', action='store_true')
    parsers.add_argument('--no_model_ema', action='store_false', dest='model_ema')
    parsers.add_argument('--model_ema_decay', type=float, default=0.99996, help='')
    parsers.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # optimizer parameters
    parsers.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parsers.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parsers.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parsers.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parsers.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    # learning rate schedule parameters
    parsers.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parsers.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 5e-4)')
    parsers.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parsers.add_argument('--lr_noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parsers.add_argument('--lr_noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parsers.add_argument('--warmup_lr', type=float, default=1e-7, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parsers.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parsers.add_argument('--decay_epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parsers.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parsers.add_argument('--cooldown_epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parsers.add_argument('--patience_epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parsers.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # augmentation parameters
    parsers.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    parsers.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + \ "(default: rand-m9-mstd0.5-inc1)')
    parsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parsers.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parsers.add_argument('--repeated_aug', action='store_true')
    parsers.add_argument('--no_repeated_aug', action='store_false', dest='repeated_aug')
    parsers.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parsers.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # mixup parameters
    parsers.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parsers.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parsers.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parsers.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parsers.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parsers.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # train parameters
    parsers.add_argument('--seed', default=2319, type=int)
    parsers.add_argument('--apex-opt-level', default='O1')
    parsers.add_argument('--throughput', action='store_true')
    parsers.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parsers.add_argument('--init_method', type=str, help='init_method')
    parsers.add_argument('--local_rank', type=int, default=0)
    parsers.add_argument('--save_epoch', default=1, type=int)
    parsers.add_argument('--print_freq', default=400, type=int)

    # datasets parameters
    parsers.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parsers.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parsers.set_defaults(pin_mem=True)
    parsers.add_argument('--debug', action='store_true')
    parsers.add_argument('--update_temperature', action='store_true')
    parsers.add_argument('--warmup_drop-path', action='store_true')
    parsers.add_argument('--warmup_drop-path-epochs', type=int, default=100)

    return parsers


def main(args):
    utils.init_distributed_mode(args)

    use_amp = None
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    else:
        print("APEX is not available")

    device = torch.device(args.device)

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True)

    val_expand = 1.0 if args.throughput else 1.5
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(val_expand * args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
        persistent_workers=True)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = build_backbone(backbone_name=args.model, num_classes=args.nb_classes)
    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    model_without_ddp = model
    if args.distributed:
        if args.apex_amp and has_apex:
            print("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    print('learning rate: ', args.lr)
    optimizer = create_optimizer(args, model)
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_opt_level)
        loss_scaler = ApexScaler()
        print('Using NVIDIA APEX AMP. Training in mixed precision.')
    else:
        loss_scaler = NativeScaler()
        print('APEX AMP not enabled. Training in default.')

    if args.throughput:
        throughput(data_loader_val, model)
        return

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    save_path = Path(args.save_path)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.model_ema:
            model_without_ddp.load_state_dict(checkpoint['model_ema'])
        else:
            model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0
    ema_max_accuracy = 0.0

    if args.warmup_drop_path:
        model_without_ddp.reset_drop_path(0.0)
    for epoch in range(args.start_epoch, num_epochs):
        if args.warmup_drop_path and epoch == args.warmup_drop_path_epochs:
            model_without_ddp.reset_drop_path(args.drop_path)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, args)

        lr_scheduler.step(epoch)

        if (epoch % args.test_freq == 0) or (epoch > args.test_epoch):
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            if model_ema is not None and not args.model_ema_force_cpu:
                ema_test_stats = evaluate(data_loader_val, model_ema.ema, device, header='EMA Test:')
                print(f"Accuracy of the EMA on the {len(dataset_val)} test images: {ema_test_stats['acc1']:.3f}%")
                ema_max_accuracy = max(ema_max_accuracy, ema_test_stats["acc1"])

            if args.save_path:
                checkpoint_paths = [save_path / 'checkpoint_last.pth']
                if max_accuracy == test_stats["acc1"]:
                    checkpoint_paths.append(save_path / 'checkpoint_best.pth')
                ema_state_dict = None
                if model_ema is not None and not args.model_ema_force_cpu:
                    if ema_max_accuracy == ema_test_stats["acc1"]:
                        checkpoint_paths.append(save_path / 'checkpoint_ema_best.pth')
                    ema_state_dict = get_state_dict(model_ema)

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': ema_state_dict,
                        'args': args,
                    }, checkpoint_path)

            if model_ema is not None:
                print(f'Max accuracy: {max_accuracy:.3f}%, Max EMA accuracy: {ema_max_accuracy:.3f}%')
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             **{f'ema_test_{k}': v for k, v in ema_test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
            else:
                print(f'Max accuracy: {max_accuracy:.3f}%')
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.save_path and utils.is_main_process():
            with (save_path / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def throughput(data_loader, model):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(10):
            model(images)
        torch.cuda.synchronize()
        print("throughput averaged with 30 times.")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        print("batch_size {} throughput {}".format(batch_size, 30 * batch_size / (tic2 - tic1)))
        return


if __name__ == '__main__':
    parsers = argparse.ArgumentParser('EViT training and evaluation script', parents=[get_args_parser()])
    args = parsers.parse_args()
    main(args)

