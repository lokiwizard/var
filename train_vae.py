import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from models.vqvae import VQVAE
from utils.data import normalize_01_into_pm1


def parse_patch_nums(pn: str) -> Tuple[int, ...]:
    return tuple(map(int, pn.replace('-', '_').split('_')))


def parse_vis_scales(vis_scales: str, num_scales: int) -> Tuple[int, ...]:
    if vis_scales.strip().lower() == 'all':
        return tuple(range(num_scales))
    indices = []
    for item in vis_scales.replace('_', ',').split(','):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx += num_scales
        if idx < 0 or idx >= num_scales:
            raise ValueError(f'可视化尺度索引越界: {item}, 总尺度数={num_scales}')
        indices.append(idx)
    return tuple(indices)


def resolve_data_path(data_path: str) -> str:
    root = Path(data_path)
    imagenet10 = root / 'imagenet-10'
    return str(imagenet10 if imagenet10.is_dir() else root)


def build_loader(args):
    data_path = resolve_data_path(args.data_path)
    mid_reso = round(args.mid_reso * args.img_size)
    transform = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip() if args.hflip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    dataset = ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    print(f'[data] root={data_path}, images={len(dataset)}, classes={len(dataset.classes)}')
    return loader


def build_vae(args, device):
    patch_nums = parse_patch_nums(args.pn)
    vae = VQVAE(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        beta=args.beta,
        using_znorm=args.using_znorm,
        quant_resi=args.quant_resi,
        share_quant_resi=args.share_quant_resi,
        v_patch_nums=patch_nums,
        test_mode=False,
    ).to(device)
    if args.eini != 0:
        vae.quantize.eini(args.eini)
    return vae


def save_ckpt(path, vae, optimizer, epoch, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'vae': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'args': vars(args),
    }, path)


def save_reconstruction(path, vae, img, max_images, scale_indices):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = min(max_images, img.shape[0])
    was_training = vae.training
    vae.eval()
    with torch.no_grad():
        recon_by_scale = vae.img_to_reconstructed_img(img[:n], last_one=False)
    if was_training:
        vae.train()

    panels = [img[:n].detach().cpu()]
    panels.extend(recon_by_scale[si].detach().cpu().clamp(-1, 1) for si in scale_indices)
    grid = torch.stack(panels, dim=1).flatten(0, 1)
    save_image(grid, path, nrow=len(panels), normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser(description='训练多尺度 VQ-VAE tokenizer')
    parser.add_argument('--data-path', default='dataset/imagenet-10')
    parser.add_argument('--out-dir', default='checkpoints/vqvae')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--mid-reso', type=float, default=1.125)
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--vis-every', type=int, default=1)
    parser.add_argument('--vis-num', type=int, default=8)
    parser.add_argument('--vis-scales', default='0,1,3,6,-1')
    parser.add_argument('--rec-loss', choices=('l1', 'mse'), default='l1')
    parser.add_argument('--vq-loss-weight', type=float, default=1.0)
    parser.add_argument('--pn', default='1_2_3_4_5_6_8_10_13_16')
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--z-channels', type=int, default=32)
    parser.add_argument('--ch', type=int, default=160)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--quant-resi', type=float, default=0.5)
    parser.add_argument('--share-quant-resi', type=int, default=4)
    parser.add_argument('--using-znorm', action='store_true')
    parser.add_argument('--eini', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    loader = build_loader(args)
    vae = build_vae(args, device)
    vis_scale_indices = parse_vis_scales(args.vis_scales, len(vae.quantize.v_patch_nums))
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')

    global_step = 0
    vae.train()
    for epoch in range(args.epochs):
        last_img = None
        pbar = tqdm(loader, desc=f'epoch {epoch + 1}/{args.epochs}', dynamic_ncols=True)
        for it, (img, _) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                rec, usages, vq_loss = vae(img, ret_usages=True)
                if args.rec_loss == 'l1':
                    rec_loss = F.l1_loss(rec, img)
                else:
                    rec_loss = F.mse_loss(rec, img)
                loss = rec_loss + args.vq_loss_weight * vq_loss

            last_img = img.detach()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                rec=f'{rec_loss.item():.4f}',
                vq=f'{vq_loss.item():.4f}',
            )
            if global_step % args.log_every == 0:
                usage_txt = ' '.join(f'{u:.1f}' for u in usages) if usages is not None else '-'
                tqdm.write(
                    f'[ep {epoch:03d} it {it:05d} step {global_step:07d}] '
                    f'loss={loss.item():.4f} rec={rec_loss.item():.4f} vq={vq_loss.item():.4f} usage={usage_txt}',
                )
            global_step += 1

        save_ckpt(os.path.join(args.out_dir, 'vae-ckpt-last.pth'), vae, optimizer, epoch + 1, global_step, args)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_ckpt(os.path.join(args.out_dir, f'vae-ckpt-ep{epoch + 1:03d}.pth'), vae, optimizer, epoch + 1, global_step, args)
        if args.vis_every > 0 and (epoch + 1) % args.vis_every == 0 and last_img is not None:
            vis_path = os.path.join(args.out_dir, f'recon_ep{epoch + 1:03d}.png')
            save_reconstruction(vis_path, vae, last_img, args.vis_num, vis_scale_indices)
            tqdm.write(f'[vis] saved {vis_path}; columns=input + scales {vis_scale_indices}')

    save_ckpt(os.path.join(args.out_dir, 'vae_ch160v4096z32_custom.pth'), vae, optimizer, args.epochs, global_step, args)


if __name__ == '__main__':
    main()
