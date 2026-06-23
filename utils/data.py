import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):  # 通过 (x*2)-1 将 [0, 1] 归一化到 [-1, 1]
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # 构建数据增强
    mid_reso = round(mid_reso * final_reso)  # 先 resize 到 mid_reso，再 crop 到 final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # Resize 会把短边缩放到 mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # Resize 会把短边缩放到 mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # 构建 ImageNet 风格目录数据集：data_path/train 和 data_path/val。
    train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
    val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
