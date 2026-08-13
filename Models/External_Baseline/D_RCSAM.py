#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""D-RCSAM training script for binary EEG seizure prediction.

The script reads pre-generated HDF5/CSV fold data, scales EEG amplitudes, trains a D-RCSAM-style network with Hugging Face Trainer, selects the best checkpoint by validation ROC-AUC, and exports validation predictions."""
import math
import numpy as np
import pandas as pd
import h5py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
import random
import warnings
try:
    from torchvision.ops import DeformConv2d as TorchvisionDeformConv2d
    _HAS_TORCHVISION_DEFORM = True
except Exception:
    TorchvisionDeformConv2d = None
    _HAS_TORCHVISION_DEFORM = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EEGDatasetVIT(Dataset):
    """Dataset wrapper for EEG windows stored as arrays of shape (17, 512)."""

    def __init__(self, data, labels):
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        label = int(self.labels[idx])
        return (torch.from_numpy(data_point), label)

def collate_fn(batch):
    """Stack EEG samples and labels into the dictionary expected by Hugging Face Trainer."""
    return {'pixel_values': torch.stack([x[0] for x in batch]), 'labels': torch.tensor([x[1] for x in batch])}

def transform_dataset(data, labels):
    """Convert EEG samples to float32 and scale amplitudes by 1e6."""
    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels)
    data_scaled = data * 10 ** 6
    return (data_scaled, labels)

def to_bipolar_montage(x):
    """Convert 17 referential EEG channels to 20 bipolar derivations.

This helper is retained for compatibility but is not used by the current D-RCSAM pipeline."""
    idx_a = torch.tensor([9, 11, 3, 1, 8, 10, 2, 0, 3, 16, 14, 15, 9, 13, 16, 5, 8, 12, 15, 4], device=x.device)
    idx_b = torch.tensor([11, 3, 1, 7, 10, 2, 0, 6, 16, 14, 15, 2, 13, 16, 5, 7, 12, 15, 4, 6], device=x.device)
    y = x.index_select(1, idx_a) - x.index_select(1, idx_b)
    return y

def _pair(v):
    """Normalize a scalar or tuple argument to a two-element tuple."""
    if isinstance(v, tuple):
        return v
    return (v, v)

def _same_pad_2d(kernel_size, dilation=1):
    """Compute explicit 2-D SAME padding for the requested kernel and dilation."""
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    pad_h = dh * (kh - 1)
    pad_w = dw * (kw - 1)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)

class Conv2dSame(nn.Module):
    """2-D convolution with manual SAME padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.pad = _same_pad_2d(kernel_size, dilation=dilation)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = F.pad(x, self.pad)
        return self.conv(x)

class DeformableConv2dSame(nn.Module):
    """Deformable 2-D convolution with manual SAME padding.

If torchvision DeformConv2d is unavailable or fails at runtime, the layer falls back to a standard Conv2d with the same kernel geometry."""
    _warned_fallback = False

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, dilation=1, bias=False):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.dilation = _pair(dilation)
        self.pad = _same_pad_2d(self.kernel_size, dilation=self.dilation)
        kh, kw = self.kernel_size
        self.use_deform = _HAS_TORCHVISION_DEFORM
        self.offset_conv = nn.Conv2d(in_channels=in_channels, out_channels=2 * kh * kw, kernel_size=self.kernel_size, stride=stride, padding=0, dilation=self.dilation, bias=True)
        if self.use_deform:
            self.deform = TorchvisionDeformConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=stride, padding=0, dilation=self.dilation, bias=bias)
            self.fallback_conv = None
        else:
            self.deform = None
            self.fallback_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=stride, padding=0, dilation=self.dilation, bias=bias)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        x_pad = F.pad(x, self.pad)
        if self.use_deform:
            offset = self.offset_conv(x_pad)
            try:
                return self.deform(x_pad, offset)
            except Exception as exc:
                if not DeformableConv2dSame._warned_fallback:
                    warnings.warn(f'torchvision DeformConv2d failed at runtime; fallback to ordinary Conv2d. Error: {exc}', RuntimeWarning)
                    DeformableConv2dSame._warned_fallback = True
                if self.fallback_conv is None:
                    self.fallback_conv = nn.Conv2d(in_channels=x.size(1), out_channels=self.deform.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=0, dilation=self.dilation, bias=self.deform.bias is not None).to(x.device)
                return self.fallback_conv(x_pad)
        else:
            if not DeformableConv2dSame._warned_fallback:
                warnings.warn('torchvision.ops.DeformConv2d is not available; using ordinary Conv2d fallback.', RuntimeWarning)
                DeformableConv2dSame._warned_fallback = True
            return self.fallback_conv(x_pad)

class DepthwiseSeparableConv2dSame(nn.Module):
    """Depthwise-separable 2-D convolution followed by batch normalization and ELU."""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 5), bias=False):
        super().__init__()
        self.depthwise = Conv2dSame(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class MultiScaleDepthwiseSeparableConv(nn.Module):
    """Fuse multiple depthwise-separable convolution branches with different receptive fields."""

    def __init__(self, in_channels, out_channels, kernels=((1, 5), (1, 7), (3, 3)), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseSeparableConv2dSame(in_channels, out_channels, kernel_size=k) for k in kernels])
        self.fuse = nn.Sequential(nn.Conv2d(out_channels * len(kernels), out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ELU(inplace=True), nn.Dropout2d(dropout))

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))

class ChannelAttention(nn.Module):
    """Apply channel-wise attention using pooled descriptors and a shared MLP."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, kernel_size=1, bias=False), nn.ELU(inplace=True), nn.Conv2d(hidden, channels, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn

class SpatialAttention(nn.Module):
    """Apply spatial attention computed from channel-wise average and maximum maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = Conv2dSame(2, 1, kernel_size=(kernel_size, kernel_size), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class SoftThreshold2d(nn.Module):
    """Residual shrinkage module with a learnable channel-wise scale.

The threshold is derived from the mean absolute activation magnitude and applied with a continuous soft-thresholding operation."""

    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        z = torch.sigmoid(self.scale)
        lam = z * x.abs().mean(dim=(1, 2, 3), keepdim=True)
        threshold = lam / 2.0
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)

class DRCSAMBlock(nn.Module):
    """D-RCSAM residual block combining deformable convolution, multi-scale convolution, channel/spatial attention, shrinkage, and an identity connection."""

    def __init__(self, in_channels, out_channels, deform_kernel=(4, 1), mds_kernels=((1, 5), (1, 7), (3, 3)), dropout=0.1):
        super().__init__()
        self.private_deform = nn.Sequential(DeformableConv2dSame(in_channels, out_channels, kernel_size=deform_kernel, bias=False), nn.BatchNorm2d(out_channels), nn.ELU(inplace=True))
        self.mds = MultiScaleDepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernels=mds_kernels, dropout=dropout)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.shrinkage = SoftThreshold2d(out_channels)
        if in_channels != out_channels:
            self.identity = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.identity = nn.Identity()
        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_act = nn.ELU(inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        y = self.private_deform(x)
        y = self.mds(y)
        y = self.channel_attention(y)
        y = self.spatial_attention(y)
        y = self.shrinkage(y)
        y = self.out_norm(y + identity)
        return self.out_act(y)

class DRCSAMBackbone(nn.Module):
    """D-RCSAM backbone adapted to EEG windows of shape (B, 17, 512).

Three 2-D D-RCSAM blocks extract spatiotemporal features. The channel-height dimension is then averaged, followed by temporal 1-D convolution, global pooling, and classification."""

    def __init__(self, n_classes: int, n_chans: int=17, n_times: int=512, F1: int=8, F2: int=8, F3: int=8, F4: int=64, dropout: float=0.1):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.block1 = DRCSAMBlock(in_channels=1, out_channels=F1, deform_kernel=(4, 1), mds_kernels=((1, 5), (1, 7), (3, 3)), dropout=dropout)
        self.block2 = DRCSAMBlock(in_channels=F1, out_channels=F2, deform_kernel=(3, 3), mds_kernels=((1, 5), (1, 7), (3, 3)), dropout=dropout)
        self.block3 = DRCSAMBlock(in_channels=F2, out_channels=F3, deform_kernel=(3, 3), mds_kernels=((1, 5), (1, 7), (3, 3)), dropout=dropout)
        self.final_2d = nn.Sequential(nn.Conv2d(F3, F4, kernel_size=1, bias=False), nn.BatchNorm2d(F4), nn.ELU(inplace=True), nn.Dropout2d(dropout))
        self.temporal_conv = nn.Sequential(nn.Conv1d(F4, F4, kernel_size=7, padding=3, bias=False), nn.BatchNorm1d(F4), nn.ELU(inplace=True), nn.Dropout(dropout))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(F4, F4 // 2), nn.ELU(inplace=True), nn.Dropout(dropout), nn.Linear(F4 // 2, n_classes))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Initialize convolutional/linear layers with Kaiming normal weights and batch norms to identity."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a fixed-length feature vector from an EEG batch."""
        B, C, T = x.shape
        if C != self.n_chans or T != self.n_times:
            raise ValueError(f'Expected input of shape (B, {self.n_chans}, {self.n_times}), got {tuple(x.shape)}')
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_2d(x)
        x = x.mean(dim=2)
        x = self.temporal_conv(x)
        x = self.global_pool(x).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        logits = self.classifier(feat)
        return logits

class DRCSAMEEGModel(nn.Module):
    """Hugging Face Trainer-compatible wrapper around the D-RCSAM backbone."""

    def __init__(self, num_labels: int=2):
        super().__init__()
        self.backbone = DRCSAMBackbone(n_classes=num_labels, n_chans=17, n_times=512, F1=8, F2=8, F3=8, F4=64, dropout=0.1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values=None, labels=None, **kwargs):
        x = pixel_values
        if x is None:
            raise ValueError('pixel_values cannot be None')
        logits = self.backbone(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

def load_model():
    """Instantiate the binary D-RCSAM EEG model on the active device."""
    model = DRCSAMEEGModel(num_labels=2)
    return model.to(device)

def count_parameters(model):
    """Return total and trainable parameter counts."""
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (total_params, trainable_params)

def Load_data(training_data_path, training_labels_path, validation_data_path, validation_labels_path):
    """Load training and validation arrays from HDF5 files and labels from CSV files.

The validation split is read from data_test.hdf5 and label_test.csv to preserve the existing fold-generation convention."""
    print('Reading data ...')
    f_train = h5py.File(training_data_path, 'r')
    X_train_arr = np.asarray(f_train['tracings'])
    f_train.close()
    y_train = pd.read_csv(training_labels_path, header=None, index_col=None).values.squeeze()
    print('Training values shape:', X_train_arr.shape)
    print('Training labels shape:', y_train.shape)
    f_valid = h5py.File(validation_data_path, 'r')
    X_valid_arr = np.asarray(f_valid['tracings'])
    f_valid.close()
    y_valid = pd.read_csv(validation_labels_path, header=None, index_col=None).values.squeeze()
    print('Validation values shape:', X_valid_arr.shape)
    print('Validation labels shape:', y_valid.shape)
    print('Done.')
    return (X_train_arr, y_train, X_valid_arr, y_valid)

def get_pred_prob(trainer, dataset):
    """Run inference and return class-1 probabilities, predicted labels, and ground-truth labels."""
    logits_true_labels = trainer.predict(dataset)
    logits = torch.tensor(logits_true_labels[0])
    preds = torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    true_labels = logits_true_labels[1]
    return (probs, preds.cpu().numpy(), true_labels)

def compute_metrics(pred):
    """Compute accuracy, sensitivity, specificity, and ROC-AUC."""
    labels = pred.label_ids
    logits = pred.predictions
    predictions = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    acc = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.0
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0.0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.0
    eval_dic = {'accuracy': acc, 'Sensitivity': sensitivity, 'Specificity': specificity, 'roc_auc': roc_auc}
    return eval_dic

def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train D-RCSAM-adapted model with specified data, save model checkpoint and probabilities')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_output_dir', type=str, default='./finetuned_model')
    parser.add_argument('--results_output_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    data_dir = args.data_dir
    model_output_dir = args.model_output_dir
    results_output_dir = args.results_output_dir
    training_data_path = os.path.join(data_dir, 'data_train.hdf5')
    training_labels_path = os.path.join(data_dir, 'label_train.csv')
    validation_data_path = os.path.join(data_dir, 'data_test.hdf5')
    validation_labels_path = os.path.join(data_dir, 'label_test.csv')
    X_train, Y_train, X_val, Y_val = Load_data(training_data_path, training_labels_path, validation_data_path, validation_labels_path)
    print('Transforming data and labels...')
    transformed_train_data, transformed_train_labels = transform_dataset(X_train, Y_train)
    transformed_val_data, transformed_val_labels = transform_dataset(X_val, Y_val)
    print(f'Transformed train data shape: {transformed_train_data.shape}')
    print(f'Transformed train labels shape: {transformed_train_labels.shape}')
    print(f'Transformed val data shape: {transformed_val_data.shape}')
    print(f'Transformed val labels shape: {transformed_val_labels.shape}')
    model = load_model()
    total_params, trainable_params = count_parameters(model)
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')
    train_dataset = EEGDatasetVIT(transformed_train_data, transformed_train_labels)
    valid_dataset = EEGDatasetVIT(transformed_val_data, transformed_val_labels)
    lr = 0.0001
    epoch = 10
    warmup_r = 0.01
    batch_size = 32
    training_args = TrainingArguments(output_dir=model_output_dir, evaluation_strategy='epoch', save_strategy='epoch', learning_rate=lr, num_train_epochs=epoch, warmup_ratio=warmup_r, weight_decay=0.01, push_to_hub=False, logging_steps=10, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, load_best_model_at_end=True, save_total_limit=1, metric_for_best_model='roc_auc', greater_is_better=True, seed=args.seed)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
    trainer.train()
    print('Best checkpoint (based on val roc_auc):', trainer.state.best_model_checkpoint)
    os.makedirs(results_output_dir, exist_ok=True)
    pred_out = trainer.predict(valid_dataset)
    logits = pred_out.predictions
    y_true = pred_out.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    prob0 = probs[:, 0]
    prob1 = probs[:, 1]
    y_pred = np.argmax(logits, axis=1)
    df = pd.DataFrame({'index': np.arange(len(y_true)), 'y_true': y_true.astype(int), 'y_pred': y_pred.astype(int), 'prob0': prob0, 'prob1': prob1, 'logit0': logits[:, 0], 'logit1': logits[:, 1]})
    csv_path = os.path.join(results_output_dir, f'val_best_predictions_seed{args.seed}.csv')
    df.to_csv(csv_path, index=False)
    print('Saved val best predictions to:', csv_path)
    np.save(os.path.join(results_output_dir, f'val_best_logits_seed{args.seed}.npy'), logits)
    np.save(os.path.join(results_output_dir, f'val_best_probs_seed{args.seed}.npy'), probs)
    np.save(os.path.join(results_output_dir, f'val_best_ytrue_seed{args.seed}.npy'), y_true)
