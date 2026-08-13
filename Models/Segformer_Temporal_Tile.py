#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CHANNELS = 17
NUM_TIMEPOINTS = 512
NUM_LABELS = 2
SIGNAL_SCALE = 1e6

BACKBONE_NAME = "nvidia/mit-b0"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_RATIO = 0.01
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01

# Longitudinal bipolar montage defined over the 17 input channels:
# 0:T6, 1:T5, 2:T4, 3:T3, 4:P4, 5:P3, 6:O2, 7:O1,
# 8:FP2, 9:FP1, 10:F8, 11:F7, 12:F4, 13:F3, 14:CZ, 15:C4, 16:C3.
BIPOLAR_SOURCE_A = (
    9, 11, 3, 1, 8, 10, 2, 0, 3, 16,
    14, 15, 9, 13, 16, 5, 8, 12, 15, 4,
)
BIPOLAR_SOURCE_B = (
    11, 3, 1, 7, 10, 2, 0, 6, 16, 14,
    15, 2, 13, 16, 5, 7, 12, 15, 4, 6,
)


class EEGDatasetVIT(Dataset):
    """Torch dataset for EEG windows stored as (17, 512)."""

    def __init__(self, data, labels):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), int(self.labels[index])


def collate_fn(batch):
    """Assemble EEG samples into the dictionary expected by Trainer."""
    return {
        "pixel_values": torch.stack([sample for sample, _ in batch]),
        "labels": torch.tensor([label for _, label in batch]),
    }


def transform_dataset(data, labels):
    """Convert EEG arrays to float32 and scale amplitudes by 1e6."""
    data = np.asarray(data, dtype=np.float32) * SIGNAL_SCALE
    labels = np.asarray(labels)
    return data, labels


def to_bipolar_montage(x):
    """
    Convert 17 referential channels to 20 longitudinal bipolar derivations.

    Input:
        x: (B, 17, 512)
    Output:
        y: (B, 20, 512)
    """
    idx_a = torch.tensor(BIPOLAR_SOURCE_A, device=x.device)
    idx_b = torch.tensor(BIPOLAR_SOURCE_B, device=x.device)
    return x.index_select(1, idx_a) - x.index_select(1, idx_b)


class MITB0EEGModel(nn.Module):
    """Adapt bipolar EEG windows to a three-channel SegFormer input image."""

    def __init__(self, num_labels: int = NUM_LABELS):
        super().__init__()

        label2id = {0: 0, 1: 1}
        id2label = {0: 0, 1: 1}

        self.backbone = AutoModelForImageClassification.from_pretrained(
            BACKBONE_NAME,
            label2id=label2id,
            id2label=id2label,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self):
        """
        Train only the first patch embedding, encoder layer norm,
        and classification head.
        """
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = (
                "segformer.encoder.patch_embeddings.0" in name
                or "segformer.encoder.layer_norm" in name
                or "classifier" in name
            )

    def forward(self, pixel_values=None, labels=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values must be provided.")

        # (B, 17, 512) -> (B, 20, 512)
        x = to_bipolar_montage(pixel_values)

        # Place time on the vertical axis:
        # (B, 20, 512) -> (B, 512, 20)
        x = x.transpose(1, 2)

        # Repeat all 20 bipolar channels 25 times:
        # (B, 512, 20) -> (B, 512, 500)
        tiled = x.repeat(1, 1, 25)

        # Append the final 12 bipolar channels to obtain width 512.
        # (B, 512, 500) + (B, 512, 12) -> (B, 512, 512)
        image = torch.cat([tiled, x[:, :, -12:]], dim=2)

        # Replicate the single EEG image across RGB channels.
        image = image.unsqueeze(1).repeat(1, 3, 1, 1)

        return self.backbone(
            pixel_values=image,
            labels=labels,
            **kwargs,
        )


def load_model():
    """Instantiate the EEG-adapted SegFormer model."""
    return MITB0EEGModel(num_labels=NUM_LABELS).to(DEVICE)


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _read_hdf5(file_path):
    """Load the 'tracings' dataset from an HDF5 file."""
    with h5py.File(file_path, "r") as handle:
        return np.asarray(handle["tracings"])


def _read_labels(file_path):
    """Load a one-column CSV label file."""
    return pd.read_csv(
        file_path,
        header=None,
        index_col=None,
    ).values.squeeze()


def load_data(
    training_data_path,
    training_labels_path,
    validation_data_path,
    validation_labels_path,
):
    """Load training and validation EEG arrays and labels."""
    print("Reading data ...")

    x_train = _read_hdf5(training_data_path)
    y_train = _read_labels(training_labels_path)
    print("Training values shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)

    x_valid = _read_hdf5(validation_data_path)
    y_valid = _read_labels(validation_labels_path)
    print("Validation values shape:", x_valid.shape)
    print("Validation labels shape:", y_valid.shape)

    print("Done.")
    return x_train, y_train, x_valid, y_valid


def get_pred_prob(trainer, dataset):
    """Return class-1 probabilities, predicted labels, and ground-truth labels."""
    prediction_output = trainer.predict(dataset)
    logits = torch.tensor(prediction_output[0])

    predictions = torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    true_labels = prediction_output[1]

    return probabilities, predictions.cpu().numpy(), true_labels


def compute_metrics(prediction):
    """Compute accuracy, sensitivity, specificity, and ROC-AUC."""
    labels = prediction.label_ids
    logits = prediction.predictions

    predicted_labels = np.argmax(logits, axis=1)
    probabilities = (
        torch.softmax(torch.tensor(logits), dim=-1)[:, 1]
        .cpu()
        .numpy()
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predicted_labels,
    ).ravel()

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc = roc_auc_score(labels, probabilities)

    return {
        "accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "roc_auc": roc_auc,
    }


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_argument_parser():
    """Define command-line arguments for model training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the original SegFormer EEG representation and select "
            "the checkpoint with the highest validation ROC-AUC."
        )
    )
    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="./finetuned_model",
    )
    parser.add_argument(
        "--results_output_dir",
        type=str,
        default="./results",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_training_args(model_output_dir, seed):
    """Create the Hugging Face training configuration."""
    return TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        push_to_hub=False,
        logging_steps=10,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        seed=seed,
    )


def main():
    args = build_argument_parser().parse_args()
    set_seed(args.seed)

    x_train, y_train, x_valid, y_valid = load_data(
        os.path.join(args.data_dir, "data_train.hdf5"),
        os.path.join(args.data_dir, "label_train.csv"),
        os.path.join(args.data_dir, "data_test.hdf5"),
        os.path.join(args.data_dir, "label_test.csv"),
    )

    print("Transforming data and labels ...")
    x_train, y_train = transform_dataset(x_train, y_train)
    x_valid, y_valid = transform_dataset(x_valid, y_valid)

    print("Transformed train data shape:", x_train.shape)
    print("Transformed train labels shape:", y_train.shape)
    print("Transformed val data shape:", x_valid.shape)
    print("Transformed val labels shape:", y_valid.shape)

    model = load_model()

    total_params, trainable_params = count_parameters(model)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    trainer = Trainer(
        model=model,
        args=build_training_args(args.model_output_dir, args.seed),
        train_dataset=EEGDatasetVIT(x_train, y_train),
        eval_dataset=EEGDatasetVIT(x_valid, y_valid),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(
        "Best checkpoint (based on val roc_auc):",
        trainer.state.best_model_checkpoint,
    )


if __name__ == "__main__":
    main()
