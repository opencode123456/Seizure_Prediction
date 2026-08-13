#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.optimizer import AcceleratedOptimizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments


# Compatibility patch for environments where AcceleratedOptimizer delegates
# train()/eval() to optimizers that may not implement these methods.
if not hasattr(nn, "cross_entropy"):
    nn.cross_entropy = F.cross_entropy


def _safe_optimizer_train(self):
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        return self.optimizer.train()
    return self


def _safe_optimizer_eval(self):
    if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
        return self.optimizer.eval()
    return self


AcceleratedOptimizer.train = _safe_optimizer_train
AcceleratedOptimizer.eval = _safe_optimizer_eval


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CHANNELS = 17
NUM_TIMEPOINTS = 512
NUM_LABELS = 2
SIGNAL_SCALE = 1e6

BACKBONE_NAME = "nvidia/MambaVision-T-1K"
PROJECTED_CHANNELS = 256

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_RATIO = 0.01
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01


class EEGDatasetVIT(Dataset):
    """Torch dataset for EEG windows stored as (channels, timepoints)."""

    def __init__(self, data, labels):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.from_numpy(self.data[index])
        label = int(self.labels[index])
        return sample, label


def collate_fn(batch):
    """Assemble EEG samples into the input format expected by Trainer."""
    return {
        "pixel_values": torch.stack([sample for sample, _ in batch]),
        "labels": torch.tensor([label for _, label in batch]),
    }


def transform_dataset(data, labels):
    """Convert arrays to float32 and scale EEG amplitudes to microvolt range."""
    data = np.asarray(data, dtype=np.float32) * SIGNAL_SCALE
    labels = np.asarray(labels)
    return data, labels


def _replace_classification_head(backbone, num_labels):
    """Replace the pretrained ImageNet head with a binary classification head."""
    if not (
        hasattr(backbone, "model")
        and hasattr(backbone.model, "head")
    ):
        raise RuntimeError(
            "MambaVision backbone does not expose model.head; "
            "the classification head cannot be replaced."
        )

    in_features = backbone.model.head.in_features
    backbone.model.head = nn.Linear(in_features, num_labels)


def _unfreeze_classification_head(backbone):
    """Enable gradients for the classification head."""
    if hasattr(backbone, "classifier"):
        for parameter in backbone.classifier.parameters():
            parameter.requires_grad = True
        return

    if hasattr(backbone, "head"):
        for parameter in backbone.head.parameters():
            parameter.requires_grad = True
        return

    if hasattr(backbone, "model") and hasattr(backbone.model, "head"):
        for parameter in backbone.model.head.parameters():
            parameter.requires_grad = True
        return

    for name, parameter in backbone.named_parameters():
        lower_name = name.lower()
        if "classifier" in lower_name or ".head" in lower_name:
            parameter.requires_grad = True


def _unfreeze_patch_embedding(backbone):
    """Enable gradients for the earliest patch-embedding/stem module."""
    if hasattr(backbone, "model") and hasattr(backbone.model, "patch_embed"):
        for parameter in backbone.model.patch_embed.parameters():
            parameter.requires_grad = True
        return

    if hasattr(backbone, "patch_embed"):
        for parameter in backbone.patch_embed.parameters():
            parameter.requires_grad = True
        return

    for name, parameter in backbone.named_parameters():
        if "patch_embed" in name:
            parameter.requires_grad = True


class MITB0EEGModel(nn.Module):
    """
    Adapt 17-channel EEG windows to a MambaVision image classifier.

    The spatial projection maps each time point from 17 EEG channels to
    256 learned features. The projected feature map is then
    replicated across three image channels before being passed to MambaVision.
    """

    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        backbone_name: str = BACKBONE_NAME,
    ):
        super().__init__()

        self.spatial_linear = nn.Linear(
            NUM_CHANNELS,
            PROJECTED_CHANNELS,
            bias=True,
        )

        label2id = {0: 0, 1: 1}
        id2label = {0: 0, 1: 1}

        self.backbone = AutoModelForImageClassification.from_pretrained(
            backbone_name,
            label2id=label2id,
            id2label=id2label,
            num_labels=num_labels,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )

        _replace_classification_head(self.backbone, num_labels)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.spatial_linear.parameters():
            parameter.requires_grad = True

        _unfreeze_classification_head(self.backbone)
        _unfreeze_patch_embedding(self.backbone)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values must be provided.")

        batch_size, channels, timepoints = pixel_values.shape
        if channels != NUM_CHANNELS or timepoints != NUM_TIMEPOINTS:
            raise ValueError(
                "Expected pixel_values with shape "
                f"(B, {NUM_CHANNELS}, {NUM_TIMEPOINTS}), "
                f"got {tuple(pixel_values.shape)}."
            )

        # (B, 17, 512) -> (B, 512, 17)
        x = pixel_values.permute(0, 2, 1)

        # (B, 512, 17) -> (B, 512, PROJECTED_CHANNELS)
        x = self.spatial_linear(x)

        # (B, 512, PROJECTED_CHANNELS)
        # -> (B, PROJECTED_CHANNELS, 512)
        x = x.permute(0, 2, 1)

        # Treat the projected EEG map as a grayscale image and replicate it
        # across three channels for the pretrained image backbone.
        image = x.unsqueeze(1).repeat(1, 3, 1, 1)

        outputs = self.backbone(tensor=image)
        logits = (
            outputs["logits"]
            if isinstance(outputs, dict)
            else outputs.logits
        )

        if labels is None:
            return {"logits": logits}

        labels = labels.view(-1).long()
        loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


def load_model():
    """Instantiate the EEG-adapted MambaVision model on the active device."""
    model = MITB0EEGModel(
        num_labels=NUM_LABELS,
        backbone_name=BACKBONE_NAME,
    )
    return model.to(DEVICE)


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
    """Load a one-column label CSV as a one-dimensional NumPy array."""
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

    predictions = np.argmax(logits, axis=1)
    print("unique labels:", np.unique(labels))
    print("unique preds:", np.unique(predictions))

    probabilities = (
        torch.softmax(torch.tensor(logits), dim=-1)[:, 1]
        .cpu()
        .numpy()
    )

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

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
            "Train MambaVision on EEG windows and select the checkpoint "
            "with the highest validation ROC-AUC."
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

    training_data_path = os.path.join(
        args.data_dir,
        "data_train.hdf5",
    )
    training_labels_path = os.path.join(
        args.data_dir,
        "label_train.csv",
    )
    validation_data_path = os.path.join(
        args.data_dir,
        "data_test.hdf5",
    )
    validation_labels_path = os.path.join(
        args.data_dir,
        "label_test.csv",
    )

    x_train, y_train, x_valid, y_valid = load_data(
        training_data_path,
        training_labels_path,
        validation_data_path,
        validation_labels_path,
    )

    print("Transforming data and labels ...")
    x_train, y_train = transform_dataset(x_train, y_train)
    x_valid, y_valid = transform_dataset(x_valid, y_valid)

    print("Transformed train data shape:", x_train.shape)
    print("Transformed train labels shape:", y_train.shape)
    print("Transformed val data shape:", x_valid.shape)
    print("Transformed val labels shape:", y_valid.shape)

    model = load_model()

    print("config.num_labels =", model.backbone.config.num_labels)
    print("head out_features =", model.backbone.model.head.out_features)

    total_params, trainable_params = count_parameters(model)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    train_dataset = EEGDatasetVIT(x_train, y_train)
    valid_dataset = EEGDatasetVIT(x_valid, y_valid)

    trainer = Trainer(
        model=model,
        args=build_training_args(args.model_output_dir, args.seed),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
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
