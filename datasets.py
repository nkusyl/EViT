import os
import json
from PIL import Image
import torch
from timm.data import create_transform
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import ImageFolder, default_loader


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(args.datasets_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(args.datasets_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.dataset_name == 'IMNET':
        root = os.path.join(args.datasets_path, 'train_images' if is_train else 'val_images')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.dataset_name == 'FLOWER':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 102
    else:
        raise NotImplementedError

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

