# %%

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
import math
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import zipfile
import shutil
import glob
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.checkpoint import checkpoint

os.makedirs("./outputs", exist_ok=True)

# %%
# Axial Mamba Module

class AxialMambaBlock(nn.Module):
    """
    Axial Mamba block: applies Mamba mixing along height and width axes separately.
    """
    def __init__(self, in_channels, out_channels, args):
        super(AxialMambaBlock, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Projections for height-axis Mamba
        self.height_down = nn.Conv2d(in_channels, args.model_input_dims, kernel_size=1, bias=False)
        self.height_mamba = MambaBlock(args)
        self.height_up = nn.Conv2d(args.model_input_dims, out_channels, kernel_size=1, bias=False)
        self.height_norm = nn.BatchNorm2d(out_channels)

        # Projections for width-axis Mamba
        self.width_down = nn.Conv2d(in_channels, args.model_input_dims, kernel_size=1, bias=False)
        self.width_mamba = MambaBlock(args)
        self.width_up = nn.Conv2d(args.model_input_dims, out_channels, kernel_size=1, bias=False)
        self.width_norm = nn.BatchNorm2d(out_channels)

        self.skip_proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        b, c, h, w = x.shape

        h_proj = self.height_down(x)
        h_seq = rearrange(h_proj, 'b d h w -> (b w) h d')
        h_seq = self.height_mamba(h_seq)
        h_mixed = rearrange(h_seq, '(b w) h d -> b d h w', b=b, w=w)
        h_out = self.height_up(h_mixed)
        h_out = self.height_norm(h_out)

        w_proj = self.width_down(x)
        w_seq = rearrange(w_proj, 'b d h w -> (b h) w d')
        w_seq = self.width_mamba(w_seq)
        w_mixed = rearrange(w_seq, '(b h) w d -> b d h w', b=b, h=h)
        w_out = self.width_up(w_mixed)
        w_out = self.width_norm(w_out)

        skip = self.skip_proj(x)
        out = h_out + w_out + skip
        return out

# %%
# Basic Double Convolution Block

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# %%

def selective_scan(u, delta, A, B, C, D):
    A = torch.clamp(A, min=-5.0, max=5.0)
    dA = torch.einsum('bld,dn->bldn', delta, A)
    dB_u = torch.einsum('bld,bld,bln->bldn', delta, u, B)
    dA_cumsum = torch.cat([dA[:, 1:], torch.zeros_like(dA[:, :1])], dim=1)
    dA_cumsum = torch.flip(dA_cumsum, dims=[1])
    dA_cumsum = torch.cumsum(dA_cumsum, dim=1)
    dA_cumsum = torch.clamp(dA_cumsum, max=15.0)
    dA_cumsum = torch.exp(dA_cumsum)
    dA_cumsum = torch.flip(dA_cumsum, dims=[1])
    x = dB_u * dA_cumsum
    x = torch.cumsum(x, dim=1) / (dA_cumsum + 1e-6)
    y = torch.einsum('bldn,bln->bld', x, C)
    return y + u * D


class MambaBlock(nn.Module):
    def __init__(self, args):
        super(MambaBlock, self).__init__()
        self.args = args
        self.in_proj = nn.Linear(args.model_input_dims, args.model_internal_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            args.model_internal_dim,
            args.model_internal_dim,
            kernel_size=args.conv_kernel_size,
            padding=args.conv_kernel_size - 1,
            groups=args.model_internal_dim
        )
        self.x_proj = nn.Linear(args.model_internal_dim, args.delta_t_rank + args.model_states * 2, bias=False)
        self.delta_proj = nn.Linear(args.delta_t_rank, args.model_internal_dim)

        A_vals = torch.arange(1, args.model_states + 1).float() / args.model_states * 3
        self.A_log = nn.Parameter(torch.log(repeat(A_vals, 'n -> d n', d=args.model_internal_dim)))
        self.D = nn.Parameter(torch.ones(args.model_internal_dim))
        self.out_proj = nn.Linear(args.model_internal_dim, args.model_input_dims, bias=args.dense_use_bias)

    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        b, l, d = x.shape
        x_and_res = self.in_proj(x)
        x1, res = x_and_res.chunk(2, dim=-1)

        x1 = rearrange(x1, 'b l d -> b d l')
        x1 = self.conv1d(x1)[..., :l]
        x1 = rearrange(x1, 'b d l -> b l d')
        x1 = F.silu(x1)

        A = -torch.exp(torch.clamp(self.A_log, min=-5, max=5))
        D = self.D
        x_dbl = self.x_proj(x1)
        delta, B, C = torch.split(
            x_dbl,
            [self.args.delta_t_rank, self.args.model_states, self.args.model_states],
            dim=-1
        )
        delta = F.softplus(self.delta_proj(delta))

        y = selective_scan(x1, delta, A, B, C, D)
        y = y * F.silu(res)
        return self.out_proj(y)

# %%

class ResidualBlock(nn.Module):
    def __init__(self, args):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(args.model_input_dims)
        self.mixer = MambaBlock(args)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.norm2 = nn.LayerNorm(args.model_input_dims)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = self.dropout(x)
        x = residual + x
        return self.norm2(x)

class ModelArgs:
    def __init__(self):
        self.model_input_dims = 96
        self.model_states = 96
        self.projection_expand_factor = 2
        self.conv_kernel_size = 4
        self.dense_use_bias = False
        self.dropout_rate = 0.2
        self.num_layers = 6
        self.num_classes = 2
        self.model_internal_dim = self.projection_expand_factor * self.model_input_dims
        self.delta_t_rank = math.ceil(self.model_input_dims / 16)

class IoULoss(nn.Module):

    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (2, 3)
        intersection = (inputs_soft * targets_one_hot).sum(dim=dims)
        union = (inputs_soft + targets_one_hot - inputs_soft * targets_one_hot).sum(dim=dims) + self.eps

        iou = (intersection + self.eps) / union
        iou_loss = 1.0 - iou
        return iou_loss.mean()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        cardinality = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):

        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        focal_term = ((1 - pt) ** self.gamma) * logpt
        loss = -focal_term
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (2, 3)
        TP = (inputs_soft * targets_one_hot).sum(dim=dims)
        FP = (inputs_soft * (1 - targets_one_hot)).sum(dim=dims)
        FN = ((1 - inputs_soft) * targets_one_hot).sum(dim=dims)

        tversky_index = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return (1.0 - tversky_index).mean()


class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (2, 3)
        intersection = (inputs_soft * targets_one_hot).sum(dim=dims)
        union = inputs_soft.sum(dim=dims) + targets_one_hot.sum(dim=dims) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return (1.0 - iou).mean()


class BoundaryLoss(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super(BoundaryLoss, self).__init__()
        self.eps = eps

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:

        m = mask.unsqueeze(1)
        inv = 1.0 - m
        eroded_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - eroded_inv
        b = m - eroded
        return b.squeeze(1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape
        probs = F.softmax(inputs, dim=1)
        gt = F.one_hot(targets, num_classes=C).permute(0,3,1,2).float()

        loss = 0.0
        for c in range(C):
            p_c = probs[:, c, :, :]
            g_c = gt[:, c, :, :]
            b_pred = self._get_boundary(p_c)
            b_gt   = self._get_boundary(g_c)
            inter = (b_pred * b_gt).sum(dim=(1,2))
            union = b_pred.sum(dim=(1,2)) + b_gt.sum(dim=(1,2))
            dice_b = (2.0 * inter + self.eps) / (union + self.eps)
            loss += (1.0 - dice_b).mean()
        return loss / C


class ComboLoss(nn.Module):
    """
    Weighted sum of BCE, Dice, Focal, Tversky, IoU, and Boundary losses.
    Loss weights are dataset-specific and detailed in the README and paper.
    """
    def __init__(self,
                 bce_weight=0.15,
                 dice_weight=0.25,
                 focal_weight=0.2,
                 tversky_weight=0.15,
                 iou_weight=0.15,
                 boundary_weight=0.1,
                 focal_gamma=2.0,
                 tversky_alpha=0.5,
                 tversky_beta=0.5):
        super(ComboLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight

        self.ce_loss       = nn.CrossEntropyLoss()
        self.dice_loss     = DiceLoss()
        self.focal_loss    = FocalLoss(gamma=focal_gamma, reduction='mean')
        self.tversky_loss  = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.iou_loss      = IoULoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, inputs, targets):
        loss_bce      = self.ce_loss(inputs, targets)
        loss_dice     = self.dice_loss(inputs, targets)
        loss_focal    = self.focal_loss(inputs, targets)
        loss_tversky  = self.tversky_loss(inputs, targets)
        loss_iou      = self.iou_loss(inputs, targets)
        loss_boundary = self.boundary_loss(inputs, targets)

        total = (
            self.bce_weight      * loss_bce
            + self.dice_weight   * loss_dice
            + self.focal_weight  * loss_focal
            + self.tversky_weight* loss_tversky
            + self.iou_weight    * loss_iou
            + self.boundary_weight * loss_boundary
        )
        return total


# %%
class DeepSupervisionLoss(nn.Module):

    def __init__(self, main_weight=0.6, deep2_weight=0.2, deep3_weight=0.2):
        super(DeepSupervisionLoss, self).__init__()
        self.main_weight = main_weight
        self.deep2_weight = deep2_weight
        self.deep3_weight = deep3_weight
        self.criterion = ComboLoss(
            bce_weight=0.2,
            dice_weight=0.3,
            focal_weight=0.2,
            tversky_weight=0.2,
            iou_weight=0.3,
            focal_gamma=2.0,
            tversky_alpha=0.5,
            tversky_beta=0.5
        )

    def forward(self, outputs, target):

        main_out, deep2, deep3, *_ = outputs
        loss_main = self.criterion(main_out, target)
        loss_deep2 = self.criterion(deep2, target)
        loss_deep3 = self.criterion(deep3, target)
        total_loss = (
            self.main_weight * loss_main +
            self.deep2_weight * loss_deep2 +
            self.deep3_weight * loss_deep3
        )
        return total_loss

# %%
# Evaluation Metrics

def calculate_iou(pred_mask, gt_mask):

    pred_mask = (pred_mask > 0).cpu().numpy().astype(bool)
    gt_mask = (gt_mask > 0).cpu().numpy().astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask):

    pred_mask = (pred_mask > 0).cpu().numpy().astype(bool)
    gt_mask = (gt_mask > 0).cpu().numpy().astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    sum_areas = pred_mask.sum() + gt_mask.sum()
    if sum_areas == 0:
        return 1.0
    return 2.0 * intersection / sum_areas


def plot_training_progress(history, epoch):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Dice')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./outputs/training_progress_epoch_{epoch}.png")
    plt.close()

# %%
# Training Function

def train_one_epoch_enhanced(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    sample_count = 0

    pbar = tqdm(dataloader, desc='Training')
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        if i == 0:
            print(f"Training batch - Images: {images.shape}, Masks: {masks.shape}")
            print(f"Masks unique values: {torch.unique(masks)}")

        outputs = model(images, return_deep=True)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        main_output = outputs[0]
        batch_size = images.size(0)
        preds = torch.argmax(main_output, dim=1)

        running_loss += loss.item() * batch_size

        batch_iou = 0.0
        batch_dice = 0.0
        for j in range(batch_size):
            iou = calculate_iou(preds[j], masks[j])
            dice = calculate_dice(preds[j], masks[j])
            batch_iou += iou
            batch_dice += dice

        running_iou += batch_iou
        running_dice += batch_dice
        sample_count += batch_size

        pbar.set_postfix({
            'loss': loss.item(),
            'iou': batch_iou / batch_size,
            'dice': batch_dice / batch_size
        })

        del outputs, loss, preds
        if i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    epoch_loss = running_loss / sample_count
    epoch_iou = running_iou / sample_count
    epoch_dice = running_dice / sample_count
    return epoch_loss, epoch_iou, epoch_dice

# Validation Function

def validate_enhanced(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images, return_deep=True)
            loss = criterion(outputs, masks)

            main_output = outputs[0]
            batch_size = images.size(0)
            preds = torch.argmax(main_output, dim=1)

            running_loss += loss.item() * batch_size
            for j in range(batch_size):
                iou = calculate_iou(preds[j], masks[j])
                dice = calculate_dice(preds[j], masks[j])
                running_iou += iou
                running_dice += dice

            sample_count += batch_size

    val_loss = running_loss / sample_count
    val_iou = running_iou / sample_count
    val_dice = running_dice / sample_count
    return val_loss, val_iou, val_dice

# Visualization Function

def visualize_results(model, dataloader, device, num_samples=3):
    import matplotlib.pyplot as plt
    model.eval()

    images, masks = next(iter(dataloader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    images = images * std + mean

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(predictions[i].cpu().numpy(), cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("./outputs/mamba_segmentation_with_axial_mamba_aspp_ppm_iou_results.png")
    plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

class MambaUNetWithAxialMambaEfficientNetB0(nn.Module):
    def __init__(self, args, in_channels=3):
        super().__init__()
        self.args = args

        #EfficientNetB0 pretrained
        backbone = efficientnet_b0(pretrained=True)

        if in_channels != 3:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                backbone.features[0][1],
                backbone.features[0][2],
            )
        else:
            self.stem = backbone.features[0]

        # Encoder stages
        self.stage1 = backbone.features[1]
        self.stage2 = backbone.features[2]
        self.stage3 = backbone.features[3]
        self.stage4 = backbone.features[4]
        self.stage5 = nn.Sequential(
            backbone.features[5],
            backbone.features[6],
            backbone.features[7],
        )

        # Axial Mamba bottleneck
        self.axial_mamba = AxialMambaBlock(320, 320, args)


        self.bridge_down = nn.Conv2d(320, args.model_input_dims, kernel_size=1)
        self.mamba_blocks = nn.Sequential(*[ResidualBlock(args) for _ in range(args.num_layers)])
        self.bridge_up = nn.Conv2d(args.model_input_dims, 320, kernel_size=1)

        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(160 + 80, 160)

        self.upconv3 = nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(80 + 40, 80)

        self.upconv2 = nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(40 + 24, 40)

        self.upconv1 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(24 + in_channels, 24)

        # Final conv and deep supervision
        self.final_conv = nn.Conv2d(24, args.num_classes, kernel_size=1)
        self.deep_sup4 = nn.Conv2d(160, args.num_classes, kernel_size=1)
        self.deep_sup3 = nn.Conv2d(80, args.num_classes, kernel_size=1)
        self.deep_sup2 = nn.Conv2d(40, args.num_classes, kernel_size=1)

    def forward(self, x, return_deep=False):
        x0 = self.stem(x)
        e1 = self.stage1(x0)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        e5 = self.stage5(e4)

        axial_out = self.axial_mamba(e5)
        bottleneck_feat = axial_out

        mamba_in = self.bridge_down(bottleneck_feat)
        b, c, h, w = mamba_in.shape
        m_in = mamba_in.permute(0, 2, 3, 1).reshape(b, h * w, c)
        m_out = self.mamba_blocks(m_in)
        m_out = m_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        mamba_out = self.bridge_up(m_out)

        bottleneck_out = self.bottleneck_refine(mamba_out + bottleneck_feat)

        d4 = self.upconv4(bottleneck_out)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))
        ds4 = self.deep_sup4(d4)

        d3 = self.upconv3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        ds3 = self.deep_sup3(d3)

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        ds2 = self.deep_sup2(d2)

        d1 = self.upconv1(d2)
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x], dim=1))

        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        if return_deep:
            ds2_up = F.interpolate(ds2, size=x.shape[2:], mode='bilinear', align_corners=False)
            ds3_up = F.interpolate(ds3, size=x.shape[2:], mode='bilinear', align_corners=False)
            ds4_up = F.interpolate(ds4, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, ds2_up, ds3_up, ds4_up

        return out





# %%

def download_and_setup_dataset(force_download=False):
    # base_path = '/kaggle/working/datasets' ##If you are using Kaggle
    base_path = os.path.expanduser("~/datasets")   # or "./datasets"
    kvasir_path = os.path.join(base_path, 'kvasir-seg')

    # kaggle_input_path = '/kaggle/input' #if you are using Kaggle
    kaggle_input_path = base_path 
    # Uncomment the next block if you are using kaggle
    # for dirname, _, _ in os.walk(kaggle_input_path):
    #     if 'kvasir-seg' in dirname.lower() and os.path.exists(os.path.join(dirname, 'images')):
    #         print(f"Found Kvasir-SEG dataset at {dirname}")
    #         return dirname

    os.makedirs(base_path, exist_ok=True)

    #If images/masks exist already and no forced download is required, return path,
    #This is similar to the check used for PolypGen, ClinicDB, and ColonDB Datasets

    if (os.path.exists(os.path.join(kvasir_path, 'images')) and
        os.path.exists(os.path.join(kvasir_path, 'masks')) and
        len(os.listdir(os.path.join(kvasir_path, 'images'))) > 0 and
        not force_download):
        print("Kvasir-SEG dataset already exists.")
        return kvasir_path

    # Direct download for Kvasir-SEG from public URL (others use kagglehub)
    # For datasets like ClinicDB, ColonDB or PolypGen, replace this section with kagglehub.dataset_download(X)
    # X = "balraj98/cvcclinicdb"(ClinicDB) ; "longvil/cvc-colondb"(ColonDB) ; "kokoroou/polypgen2021"(PolypGen)

    dataset_url = "https://datasets.simula.no/downloads/kvasir-seg.zip" #Kvasir-SEG direct download link
    # import kagglehub
    # X = "balraj98/cvcclinicdb"(
    # kagglehub.dataset_download(X)
    zip_path = os.path.join(base_path, 'kvasir-seg.zip')

    print("Downloading Kvasir-SEG dataset...")
    try:
        # response = requests.get(dataset_url, stream=True)
        response = requests.get(dataset_url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(zip_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

    print(f"Download completed, file saved to {zip_path}")

    # Extract the zip (in PolypGen and others, this step is not needed as kagglehub provides unzipped folders)
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_path)
        print("Extraction completed.")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return None

    extracted_files = glob.glob(os.path.join(base_path, "**"), recursive=True)
    print("Extracted file structure:")
    for file in extracted_files[:10]:
        print(f"  {file}")
    if len(extracted_files) > 10:
        print(f"  ... and {len(extracted_files)-10} more files")

    image_dirs = glob.glob(os.path.join(base_path, "**/images"), recursive=True)
    mask_dirs = glob.glob(os.path.join(base_path, "**/masks"), recursive=True)

    print(f"Found image directories: {image_dirs}")
    print(f"Found mask directories: {mask_dirs}")

    # Create unified 'images' and 'masks' folders under /kvasir-seg for consistency
    # In other dataset codes, this is also done to standardize directory structure

    os.makedirs(os.path.join(kvasir_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(kvasir_path, 'masks'), exist_ok=True)

    # For all datasets, src_image_dir and src_mask_dir are taken from glob results
    if image_dirs and mask_dirs:
        src_image_dir = image_dirs[0]
        src_mask_dir = mask_dirs[0]

        if src_image_dir != os.path.join(kvasir_path, 'images'):
            print(f"Moving images from {src_image_dir} to {os.path.join(kvasir_path, 'images')}")
            for img_file in os.listdir(src_image_dir):
                shutil.copy(
                    os.path.join(src_image_dir, img_file),
                    os.path.join(kvasir_path, 'images', img_file)
                )

        if src_mask_dir != os.path.join(kvasir_path, 'masks'):
            print(f"Moving masks from {src_mask_dir} to {os.path.join(kvasir_path, 'masks')}")
            for mask_file in os.listdir(src_mask_dir):
                shutil.copy(
                    os.path.join(src_mask_dir, mask_file),
                    os.path.join(kvasir_path, 'masks', mask_file)
                )

    try:
        os.remove(zip_path)
        print("Removed zip file.")
    except:
        print("Could not remove zip file.")

    # Final check to confirm setup — used in all dataset scripts
    if (os.path.exists(os.path.join(kvasir_path, 'images')) and
        os.path.exists(os.path.join(kvasir_path, 'masks')) and
        len(os.listdir(os.path.join(kvasir_path, 'images'))) > 0):
        print("Dataset setup completed successfully.")
        print(f"Found {len(os.listdir(os.path.join(kvasir_path, 'images')))} images and "
              f"{len(os.listdir(os.path.join(kvasir_path, 'masks')))} masks.")
        return kvasir_path
    else:
        print("Dataset setup failed.")
        return None

# %%

class KvasirSEGDataset(Dataset):   #Change according to the dataset used
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, augment=True):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment and split == 'train'
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

        if not os.path.exists(self.img_dir):
            raise ValueError(f"Images directory not found: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Masks directory not found: {self.mask_dir}")

        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        if not self.images:
            raise ValueError(f"No images found in {self.img_dir}")

        print(f"Sample image names: {self.images[:5]}")

        np.random.seed(42)
        indices = np.random.permutation(len(self.images))

        if split == 'train':
            self.images = [self.images[i] for i in indices[:int(0.8 * len(self.images))]]
        elif split == 'val':
            self.images = [self.images[i] for i in indices[int(0.8 * len(self.images)):int(0.9 * len(self.images))]]
        else:
            self.images = [self.images[i] for i in indices[int(0.9 * len(self.images)):]]

        print(f"Created {split} dataset with {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_candidates = [
            os.path.join(self.mask_dir, base_name + ext)
            for ext in ['.jpg', '.png', '.jpeg', '.tif']
        ]
        mask_path = next((path for path in mask_candidates if os.path.exists(path)), None)

        if not mask_path:
            mask_files = os.listdir(self.mask_dir)
            matches = [f for f in mask_files if f.startswith(base_name)]
            if matches:
                mask_path = os.path.join(self.mask_dir, matches[0])
            else:
                raise FileNotFoundError(f"No mask found for image {img_name}")

        if idx == 0:
            print(f"Image path: {img_path}")
            print(f"Mask path: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                fill = 0
                image = TF.rotate(image, angle, fill=fill)
                mask = TF.rotate(mask, angle, fill=fill)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask_array = np.array(mask)
            mask_binary = (mask_array > 0).astype(np.int64)
            mask = torch.from_numpy(mask_binary).long()

        if idx == 0:
            print(f"Image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, range: [{mask.min()}, {mask.max()}]")

        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)

        return image, mask

# %%

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    kvasir_path = download_and_setup_dataset(force_download=False)
    if not kvasir_path:
        print("Dataset setup failed. Exiting...")
        return

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        lambda x: (x > 0.5).long()
    ])

    try:
        train_dataset = KvasirSEGDataset(
            kvasir_path,
            split='train',
            transform=transform,
            target_transform=target_transform,
            augment=True
        )
        val_dataset = KvasirSEGDataset(
            kvasir_path,
            split='val',
            transform=transform,
            target_transform=target_transform,
            augment=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print("Data loaders created successfully.")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    args = ModelArgs()
    model = MambaUNetWithAxialMambaEfficientNetB0(args).to(device)

    criterion = DeepSupervisionLoss(main_weight=0.6, deep2_weight=0.2, deep3_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    num_epochs = 300
    best_iou = 0.0
    patience_counter = 0
    max_patience = 300

    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }

    print(f"Starting training for {num_epochs} epochs...")
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_iou, train_dice = train_one_epoch_enhanced(
                model, train_loader, optimizer, criterion, device, scheduler
            )
            val_loss, val_iou, val_dice = validate_enhanced(
                model, val_loader, criterion, device
            )

            history['train_loss'].append(train_loss)
            history['train_iou'].append(train_iou)
            history['train_dice'].append(train_dice)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            history['val_dice'].append(val_dice)

            print(f"Train → Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   → Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), "./outputs/best_mamba_unet_axial_mamba_aspp_ppm_iou.pth")
                print(f"Model saved with IoU: {best_iou:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_iou': best_iou,
                    'history': history,
                }, f"./outputs/checkpoint_axial_mamba_aspp_ppm_iou_epoch_{epoch+1}.pth")
                plot_training_progress(history, epoch + 1)

            if patience_counter >= max_patience:
                print(f"Early stopping after {max_patience} epochs without improvement")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
            'best_iou': best_iou if 'best_iou' in locals() else 0,
            'history': history,
        }, "./outputs/error_checkpoint_axial_mamba_aspp_ppm_iou.pth")

    # try:
    #     model.load_state_dict(torch.load("./outputs/best_mamba_unet_axial_mamba_aspp_ppm_iou.pth"))
    #     print("Loaded best model for evaluation")
    # except:
    #     print("Could not load best model, using current model")

    # visualize_results(model, val_loader, device)
    # print("Training and evaluation completed!")

if __name__ == "__main__":
    main()


# %%

args = ModelArgs()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MambaUNetWithAxialMambaEfficientNetB0(args).to(device)

# model.load_state_dict(torch.load("./outputs/best_mamba_unet_axial_mamba_aspp_ppm_iou.pth"))

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
target_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
    lambda x: (x > 0.5).long()
])
kvasir_path = download_and_setup_dataset(force_download=False)
val_dataset = KvasirSEGDataset(kvasir_path, split='val', transform=transform, target_transform=target_transform, augment=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)


# %%
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

def evaluate_precision_recall_f1(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images, return_deep=True)
            main_output = outputs[0]

            preds = torch.argmax(main_output, dim=1)
            masks = masks.squeeze(1)

            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(masks.cpu().numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return precision, recall, f1


# %%
precision, recall, f1 = evaluate_precision_recall_f1(model, val_loader, device)
print(f"Final Validation Metrics → Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


# %%
# !pip install torchinfo

# %%


from torchinfo import summary
import torch

args = ModelArgs()
model = MambaUNetWithAxialMambaEfficientNetB0(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
summary(model, input_size=(1, 3, 256, 256), device=str(device), depth=4)


# %%
# !pip install thop

# %%
from thop import profile

model.eval()
input_tensor = torch.randn(1, 3, 224, 224).to(device)

macs, params = profile(model, inputs=(input_tensor,), verbose=False)
flops = 2 * macs

print(f"Params: {params / 1e6:.2f} M")
print(f"MACs: {macs / 1e9:.2f} G")
print(f"FLOPs: {flops / 1e9:.2f} G")


# %%
import torch
import time

def measure_inference_time(model, input_size=(1, 3, 256, 256), device='cuda', repeat=100):
    model.eval().to(device)
    input_tensor = torch.randn(*input_size).to(device)

    for _ in range(10):
        _ = model(input_tensor)

    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_tensor)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / repeat * 1000
    print(f"Average Inference Time: {avg_inference_time:.3f} ms per image")

model = MambaUNetWithAxialMambaEfficientNetB0(ModelArgs())
measure_inference_time(model, input_size=(1, 3, 256, 256), device='cuda', repeat=100)



