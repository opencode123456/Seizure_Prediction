#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DSAINet training script for binary EEG seizure prediction.

The script reads pre-generated HDF5/CSV fold data, scales EEG amplitudes, trains the adapted DSAINet model with Hugging Face Trainer, selects the best checkpoint by validation ROC-AUC, and exports validation predictions."""
import math
import numpy as np
import pandas as pd
import h5py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
import random
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

This helper is retained for compatibility but is not used by the current DSAINet pipeline."""
    idx_a = torch.tensor([9, 11, 3, 1, 8, 10, 2, 0, 3, 16, 14, 15, 9, 13, 16, 5, 8, 12, 15, 4], device=x.device)
    idx_b = torch.tensor([11, 3, 1, 7, 10, 2, 0, 6, 16, 14, 15, 2, 13, 16, 5, 7, 12, 15, 4, 6], device=x.device)
    y = x.index_select(1, idx_a) - x.index_select(1, idx_b)
    return y

class PatchEmbedding(nn.Module):
    """Extract compact spatiotemporal EEG tokens with temporal convolution, grouped spatial convolution, pooling, and dropout.

Input shape: (B, 1, C, T). Output shape: (B, F2, 1, N)."""

    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=4, pooling_size2=8, dropout_rate=0.25, number_channel=17):
        super().__init__()
        f2 = D * f1
        self.f2 = f2
        self.net = nn.Sequential(nn.Conv2d(1, f1, (1, kernel_size), padding='same', bias=False), nn.BatchNorm2d(f1), nn.Conv2d(f1, f2, (number_channel, 1), groups=f1, padding='valid', bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, pooling_size1)), nn.Dropout(dropout_rate), nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, pooling_size2)), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PositionalEncoding(nn.Module):
    """Add learnable positional embeddings to token sequences of shape (B, N, E)."""

    def __init__(self, emb_size: int, length: int=512, dropout: float=0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, length, emb_size))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        return self.drop(x + self.pe[:, :n, :].to(x.device))

class ConvTimeLayer(nn.Module):
    """Temporal convolution block with depthwise convolution, grouped pointwise expansion, batch normalization, dropout, and a learnable residual scale."""

    def __init__(self, emb_size: int, kernel_size: int, expansion: int=4, dropout: float=0.1):
        super().__init__()
        self.dw = nn.Conv1d(emb_size, emb_size, kernel_size=kernel_size, padding=kernel_size // 2, groups=emb_size, bias=False)
        d_ff = expansion * emb_size
        self.pw1 = nn.Conv1d(emb_size, d_ff, 1, groups=4, bias=False)
        self.pw2 = nn.Conv1d(d_ff, emb_size, 1, groups=4, bias=False)
        self.bn = nn.BatchNorm1d(emb_size)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.dw(x))
        y = self.act(self.pw1(y))
        y = self.pw2(y)
        y = self.bn(y)
        y = self.drop(y)
        return x + self.alpha * y

class ConvTimeStack(nn.Module):
    """Apply a sequence of temporal convolution layers with different kernel sizes."""

    def __init__(self, emb_size, kernel_list, expansion=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([ConvTimeLayer(emb_size, k, expansion, dropout) for k in kernel_list])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class IntraAttnBlock(nn.Module):
    """Self-attention block operating within one temporal feature branch."""

    def __init__(self, emb_size: int, heads: int, dropout: float=0.1, ffn_expansion: int=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.drop = nn.Dropout(dropout)
        d_ff = ffn_expansion * emb_size
        self.ffn = nn.Sequential(nn.Linear(emb_size, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x

class InterAttnBlock(nn.Module):
    """Bidirectional cross-attention block that exchanges information between two branches."""

    def __init__(self, emb_size, heads, dropout=0.1, ffn_expansion=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)
        self.norm1a = nn.LayerNorm(emb_size)
        self.norm1b = nn.LayerNorm(emb_size)
        self.norm2a = nn.LayerNorm(emb_size)
        self.norm2b = nn.LayerNorm(emb_size)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.beta12 = nn.Parameter(torch.tensor(1.0))
        self.beta21 = nn.Parameter(torch.tensor(1.0))
        d_ff = ffn_expansion * emb_size
        self.ffn = nn.Sequential(nn.Linear(emb_size, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, emb_size))

    def forward(self, x1, x2):
        out1, _ = self.mha(x1, x2, x2)
        y1 = self.norm1a(x1 + self.drop_attn(self.beta12 * out1))
        y1 = self.norm1b(y1 + self.drop_ffn(self.ffn(y1)))
        out2, _ = self.mha(x2, x1, x1)
        y2 = self.norm2a(x2 + self.drop_attn(self.beta21 * out2))
        y2 = self.norm2b(y2 + self.drop_ffn(self.ffn(y2)))
        return (y1, y2)

class DSAINetBackbone(nn.Module):
    """Dual-stream DSAINet backbone adapted to EEG windows.

Patch embeddings are projected to a shared embedding space, processed by two temporal convolution branches, refined with intra- and inter-branch attention, attention-pooled, and classified."""

    def __init__(self, n_classes: int, Chans: int, Samples: int, emb_size: int=40, heads: int=4, attn_depth: int=1, attn_dropout: float=0.25, eeg1_f1: int=16, eeg1_kernel_size: int=64, eeg1_D: int=2, eeg1_pooling_size1: int=4, eeg1_pooling_size2: int=8, eeg1_dropout_rate: float=0.25, branch_1_kernels=None, branch_2_kernels=None, conv_expansion: int=4, conv_dropout: float=0.25, intra_ffn_expansion: int=2, inter_ffn_expansion: int=2, big_residual: bool=True, big_residual_learnable: bool=True, cls_dropout: float=0.25):
        super().__init__()
        if branch_1_kernels is None:
            branch_1_kernels = [11, 15]
        if branch_2_kernels is None:
            branch_2_kernels = [3, 7]
        self.emb_size = emb_size
        self.attn_depth = attn_depth
        self.big_residual = big_residual
        pos_len = Samples // (eeg1_pooling_size1 * eeg1_pooling_size2)
        self.patch = PatchEmbedding(f1=eeg1_f1, kernel_size=eeg1_kernel_size, D=eeg1_D, pooling_size1=eeg1_pooling_size1, pooling_size2=eeg1_pooling_size2, dropout_rate=eeg1_dropout_rate, number_channel=Chans)
        f2 = eeg1_f1 * eeg1_D
        self.proj = nn.Linear(f2, emb_size) if f2 != emb_size else nn.Identity()
        self.pos = PositionalEncoding(emb_size, length=pos_len, dropout=attn_dropout)
        self.branch1 = ConvTimeStack(emb_size, branch_1_kernels, expansion=conv_expansion, dropout=conv_dropout)
        self.branch2 = ConvTimeStack(emb_size, branch_2_kernels, expansion=conv_expansion, dropout=conv_dropout)
        if big_residual:
            if big_residual_learnable:
                self.alpha1 = nn.Parameter(torch.tensor(1.0))
                self.alpha2 = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer('alpha1', torch.tensor(1.0), persistent=False)
                self.register_buffer('alpha2', torch.tensor(1.0), persistent=False)
        self.intra_1 = nn.ModuleList([IntraAttnBlock(emb_size, heads, attn_dropout, intra_ffn_expansion) for _ in range(attn_depth)])
        self.intra_2 = nn.ModuleList([IntraAttnBlock(emb_size, heads, attn_dropout, intra_ffn_expansion) for _ in range(attn_depth)])
        self.inter = nn.ModuleList([InterAttnBlock(emb_size, heads, attn_dropout, inter_ffn_expansion) for _ in range(attn_depth)])
        self.token_attn = nn.Linear(emb_size, 1)
        self.classifier = nn.Sequential(nn.Dropout(cls_dropout), nn.Linear(2 * emb_size, n_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the fused dual-branch representation used by the classifier."""
        B = x.shape[0]
        fmap = self.patch(x)
        a0 = fmap.squeeze(2).transpose(1, 2)
        a0 = self.proj(a0)
        a0 = a0 * math.sqrt(self.emb_size)
        a0 = self.pos(a0)
        z0 = a0.transpose(1, 2)
        a1 = self.branch1(z0).transpose(1, 2)
        a2 = self.branch2(z0).transpose(1, 2)
        if self.big_residual:
            a1 = a1 + self.alpha1 * a0
            a2 = a2 + self.alpha2 * a0
        for i in range(self.attn_depth):
            a1 = self.intra_1[i](a1)
            a2 = self.intra_2[i](a2)
            a1, a2 = self.inter[i](a1, a2)
        x = torch.stack([a1, a2], dim=1)
        w = torch.softmax(self.token_attn(x).squeeze(-1), dim=2)
        pooled = (x * w.unsqueeze(-1)).sum(dim=2)
        feat = pooled.reshape(B, -1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.classifier(feat)

class DSAINetEEGModel(nn.Module):
    """Hugging Face Trainer-compatible wrapper for DSAINet.

Input pixel_values must have shape (B, 17, 512)."""

    def __init__(self, num_labels: int=2, chans: int=17, samples: int=512):
        super().__init__()
        self.num_labels = num_labels
        self.chans = chans
        self.samples = samples
        self.backbone = DSAINetBackbone(n_classes=num_labels, Chans=chans, Samples=samples, emb_size=40, heads=4, attn_depth=1, attn_dropout=0.25, eeg1_f1=16, eeg1_kernel_size=64, eeg1_D=2, eeg1_pooling_size1=4, eeg1_pooling_size2=8, eeg1_dropout_rate=0.25, branch_1_kernels=[11, 15], branch_2_kernels=[3, 7], conv_expansion=4, conv_dropout=0.25, intra_ffn_expansion=2, inter_ffn_expansion=2, big_residual=True, big_residual_learnable=True, cls_dropout=0.25)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values=None, labels=None, **kwargs):
        x = pixel_values
        if x is None:
            raise ValueError('pixel_values cannot be None')
        if x.ndim != 3:
            raise ValueError(f'Expected pixel_values with shape (B,17,512), got {x.shape}')
        B, C, T = x.shape
        if C != self.chans or T != self.samples:
            raise ValueError(f'Expected pixel_values of shape (B,{self.chans},{self.samples}), got {x.shape}')
        x = x.unsqueeze(1)
        logits = self.backbone(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

def load_model():
    """Instantiate the binary DSAINet EEG model on the active device."""
    model = DSAINetEEGModel(num_labels=2, chans=17, samples=512)
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
    parser = argparse.ArgumentParser(description='Train DSAINet with specified data, save model checkpoint and probabilities')
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
    print(f'Transforming data and labels...')
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
