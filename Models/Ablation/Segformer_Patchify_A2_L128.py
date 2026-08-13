#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
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
PROJECTED_CHANNELS = 32
SIGNAL_SCALE = 1e6

PATCH_SIZE = 128
HEIGHT_REPEAT = 4
WIDTH_REPEAT = 4

DEFAULT_MODEL_CHECKPOINT = "nvidia/mit-b0"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_RATIO = 0.01
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01


class EEGDataset(Dataset):
    """Dataset for EEG windows stored as arrays of shape (17, 512)."""

    def __init__(self, data, labels):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.from_numpy(self.data[index])
        label = int(self.labels[index])
        return sample, label


def collate_batch(batch):
    """Assemble EEG samples into the input dictionary expected by Trainer."""
    return {
        "pixel_values": torch.stack([sample for sample, _ in batch]),
        "labels": torch.tensor([label for _, label in batch]),
    }


def scale_eeg(data, labels):
    """Convert EEG data to float32 and scale amplitudes by 1e6."""
    data = np.asarray(data, dtype=np.float32) * SIGNAL_SCALE
    labels = np.asarray(labels)
    return data, labels


class SegFormerPatchifyModel(nn.Module):
    """
    Convert each EEG window into a 512 x 512 patchified representation.

    The 17 input channels are projected to 32 learned features at each of the
    512 time points. The resulting feature map is flattened, partitioned into
    non-overlapping patches of length 128, and tiled to 512 x 512
    before being passed to a pretrained MIT-B0 image classifier.
    """

    accepts_loss_kwargs = False

    def __init__(
        self,
        num_labels=NUM_LABELS,
        model_checkpoint=DEFAULT_MODEL_CHECKPOINT,
    ):
        super().__init__()

        self.spatial_projection = nn.Linear(
            NUM_CHANNELS,
            PROJECTED_CHANNELS,
            bias=True,
        )

        self.backbone = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            label2id={0: 0, 1: 1},
            id2label={0: 0, 1: 1},
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self):
        """
        Train the EEG projection, the first MIT patch embedding,
        the encoder layer norm, and the classification head.
        """
        for parameter in self.spatial_projection.parameters():
            parameter.requires_grad = True

        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = (
                "segformer.encoder.patch_embeddings.0" in name
                or "segformer.encoder.layer_norm" in name
                or "classifier" in name
            )

    def forward(self, pixel_values=None, labels=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values must be provided.")

        # Newer Trainer versions may pass this loss-scaling argument to custom
        # models. The underlying SegFormer classifier does not consume it.
        kwargs.pop("num_items_in_batch", None)

        batch_size, channels, timepoints = pixel_values.shape
        if channels != NUM_CHANNELS or timepoints != NUM_TIMEPOINTS:
            raise ValueError(
                "Expected pixel_values with shape "
                f"(B, {NUM_CHANNELS}, {NUM_TIMEPOINTS}), "
                f"got {tuple(pixel_values.shape)}."
            )

        # (B, 17, 512) -> (B, 512, 17)
        x = pixel_values.permute(0, 2, 1)

        # (B, 512, 17) -> (B, 512, 32)
        x = self.spatial_projection(x)

        # (B, 512, 32) -> (B, 32, 512)
        x = x.permute(0, 2, 1)

        # (B, 32, 512) -> (B, 1, 16384)
        x = x.reshape(
            batch_size,
            1,
            PROJECTED_CHANNELS * NUM_TIMEPOINTS,
        )

        # Split the flattened representation into non-overlapping patches.
        # A1/L64:  (B, 1, 16384) -> (B, 1, 256, 64)
        # A2/L128: (B, 1, 16384) -> (B, 1, 128, 128)
        patches = x.unfold(
            dimension=2,
            size=PATCH_SIZE,
            step=PATCH_SIZE,
        ).squeeze(1)

        # Tile the patch matrix to obtain a fixed 512 x 512 representation.
        image = patches.repeat(
            1,
            HEIGHT_REPEAT,
            WIDTH_REPEAT,
        )

        # Replicate the EEG representation across three image channels.
        image = image.unsqueeze(1).repeat(1, 3, 1, 1)

        return self.backbone(
            pixel_values=image,
            labels=labels,
            **kwargs,
        )


def load_model(model_checkpoint):
    """Instantiate the patchified EEG classifier on the active device."""
    model = SegFormerPatchifyModel(
        num_labels=NUM_LABELS,
        model_checkpoint=model_checkpoint,
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


def read_hdf5(file_path):
    """Load the 'tracings' dataset from an HDF5 file."""
    with h5py.File(file_path, "r") as handle:
        return np.asarray(handle["tracings"])


def read_labels(file_path):
    """Load a one-column CSV label file."""
    return pd.read_csv(
        file_path,
        header=None,
        index_col=None,
    ).values.squeeze()


def load_data(data_dir):
    """Load training and validation arrays from a fold directory."""
    train_data_path = os.path.join(data_dir, "data_train.hdf5")
    train_label_path = os.path.join(data_dir, "label_train.csv")
    val_data_path = os.path.join(data_dir, "data_test.hdf5")
    val_label_path = os.path.join(data_dir, "label_test.csv")

    required_files = [
        train_data_path,
        train_label_path,
        val_data_path,
        val_label_path,
    ]
    missing_files = [
        path for path in required_files
        if not os.path.isfile(path)
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing required input files:\n"
            + "\n".join(missing_files)
        )

    print("Reading data ...")

    x_train = read_hdf5(train_data_path)
    y_train = read_labels(train_label_path)
    print("Training values shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)

    x_val = read_hdf5(val_data_path)
    y_val = read_labels(val_label_path)
    print("Validation values shape:", x_val.shape)
    print("Validation labels shape:", y_val.shape)

    print("Done.")
    return x_train, y_train, x_val, y_val


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


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_arguments(model_output_dir, seed):
    """
    Build TrainingArguments while supporting both old and new Transformers
    names for the epoch-based evaluation strategy.
    """
    kwargs = {
        "output_dir": model_output_dir,
        "save_strategy": "epoch",
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "push_to_hub": False,
        "logging_steps": 10,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "metric_for_best_model": "roc_auc",
        "greater_is_better": True,
        "seed": seed,
        "report_to": "none",
    }

    supported = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "The installed Transformers version supports neither "
            "'eval_strategy' nor 'evaluation_strategy'."
        )

    return TrainingArguments(**kwargs)


def export_validation_predictions(
    trainer,
    validation_dataset,
    results_output_dir,
    seed,
):
    """Save validation logits, probabilities, predictions, and labels."""
    os.makedirs(results_output_dir, exist_ok=True)

    prediction_output = trainer.predict(validation_dataset)
    logits = prediction_output.predictions
    y_true = prediction_output.label_ids

    probabilities = (
        torch.softmax(torch.tensor(logits), dim=-1)
        .cpu()
        .numpy()
    )
    y_pred = np.argmax(logits, axis=1)

    prediction_table = pd.DataFrame({
        "index": np.arange(len(y_true)),
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
        "prob0": probabilities[:, 0],
        "prob1": probabilities[:, 1],
        "logit0": logits[:, 0],
        "logit1": logits[:, 1],
    })

    csv_path = os.path.join(
        results_output_dir,
        f"val_best_predictions_seed{seed}.csv",
    )
    logits_path = os.path.join(
        results_output_dir,
        f"val_best_logits_seed{seed}.npy",
    )
    probs_path = os.path.join(
        results_output_dir,
        f"val_best_probs_seed{seed}.npy",
    )
    ytrue_path = os.path.join(
        results_output_dir,
        f"val_best_ytrue_seed{seed}.npy",
    )

    prediction_table.to_csv(csv_path, index=False)
    np.save(logits_path, logits)
    np.save(probs_path, probabilities)
    np.save(ytrue_path, y_true)

    print("Saved validation predictions to:", csv_path)


def build_argument_parser():
    """Define command-line arguments for local training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train SegFormer Patchify A2_L128 locally and select "
            "the checkpoint with the highest validation ROC-AUC."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Fold directory containing data_train.hdf5, label_train.csv, "
            "data_test.hdf5, and label_test.csv."
        ),
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="./finetuned_model",
        help="Directory used to save model checkpoints.",
    )
    parser.add_argument(
        "--results_output_dir",
        type=str,
        default="./results",
        help="Directory used to save validation predictions.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT,
        help=(
            "Hugging Face model name or a local directory containing "
            "pretrained MIT-B0 weights."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()

    set_seed(args.seed)

    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.results_output_dir, exist_ok=True)

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

    x_train, y_train, x_val, y_val = load_data(args.data_dir)

    print("Transforming data and labels ...")
    x_train, y_train = scale_eeg(x_train, y_train)
    x_val, y_val = scale_eeg(x_val, y_val)

    print("Transformed train data shape:", x_train.shape)
    print("Transformed train labels shape:", y_train.shape)
    print("Transformed val data shape:", x_val.shape)
    print("Transformed val labels shape:", y_val.shape)

    model = load_model(args.model_checkpoint)

    total_params, trainable_params = count_parameters(model)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    train_dataset = EEGDataset(x_train, y_train)
    validation_dataset = EEGDataset(x_val, y_val)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(
            args.model_output_dir,
            args.seed,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_batch,
        compute_metrics=compute_metrics,
    )

    # Prevent newer Trainer versions from forwarding unsupported loss kwargs.
    if hasattr(trainer, "model_accepts_loss_kwargs"):
        trainer.model_accepts_loss_kwargs = False

    trainer.train()

    print(
        "Best checkpoint (based on val roc_auc):",
        trainer.state.best_model_checkpoint,
    )

    export_validation_predictions(
        trainer=trainer,
        validation_dataset=validation_dataset,
        results_output_dir=args.results_output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
