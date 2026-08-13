# Seizure Prediction on TUSZ

This repository contains the data split definitions and model implementations used in our EEG-based seizure prediction experiments on the **Temple University Hospital Seizure Corpus (TUSZ)**.

The repository is intended to improve the reproducibility of our experiments by providing:

* the exact five-fold experimental split lists;
* the proposed SegFormer-based Temporal-Patchify implementation;
* alternative EEG-to-image encoding configurations;
* ablation variants; and
* external baseline implementations.

> **Note:** Raw EEG recordings and original TUSZ annotations are **not redistributed** in this repository. Users should obtain TUSZ independently through the official NEDC data distribution.

---

## Repository Structure

```text
Seizure_Prediction/
├── Data/
│   ├── fold1/
│   │   ├── train.txt
│   │   └── test.txt
│   ├── fold2/
│   │   ├── train.txt
│   │   └── test.txt
│   ├── fold3/
│   │   ├── train.txt
│   │   └── test.txt
│   ├── fold4/
│   │   ├── train.txt
│   │   └── test.txt
│   └── fold5/
│       ├── train.txt
│       └── test.txt
│
└── Models/
    ├── Segformer_Patchify.py
    ├── Segformer_Temporal_Tile.py
    ├── MambaVision_Input20_512.py
    ├── MambaVision_input256_512.py
    │
    ├── Ablation/
    │   ├── Segformer_Patchify_A1_L64.py
    │   └── Segformer_Patchify_A2_L128.py
    │
    └── External_Baseline/
        ├── DSAINet.py
        └── D_RCSAM.py
```

---

## Dataset and Five-Fold Splits

Experiments are conducted using EEG recordings derived from TUSZ.

The exact five-fold experimental partitions used in our experiments are provided under:

```text
Data/fold1/
Data/fold2/
Data/fold3/
Data/fold4/
Data/fold5/
```

Each fold contains:

```text
train.txt
test.txt
```

where each line specifies one preprocessed EEG file included in the corresponding training or test partition.

For example:

```text
trn_acz_s006__2.npy
trn_auj_s003__4.npy
tst_arq_s014__441.npy
vld_hie_s007__288.npy
```

The split sizes are:

| Fold   | Training entries | Test entries | Total |
| ------ | ---------------: | -----------: | ----: |
| Fold 1 |              333 |           84 |   417 |
| Fold 2 |              333 |           84 |   417 |
| Fold 3 |              334 |           83 |   417 |
| Fold 4 |              334 |           83 |   417 |
| Fold 5 |              334 |           83 |   417 |

These text files define the **exact experimental partitions** used for five-fold evaluation and are provided to facilitate reproducibility.

### Important note on file names

Prefixes such as:

```text
trn_
vld_
tst_
```

are retained as part of the original/preprocessed file identifiers.

They should **not** be interpreted as the training or test role of a sample in the five-fold experiments.

For a given fold, the authoritative partition is determined exclusively by whether the file is listed in:

```text
Data/foldX/train.txt
```

or

```text
Data/foldX/test.txt
```

---

## Input EEG Representation

The current model implementations operate on EEG windows with the shape:

```text
17 channels × 512 time points
```

The training scripts expect preprocessed datasets stored in HDF5 format.

A typical directory supplied through `--data_dir` should contain:

```text
data_train.hdf5
label_train.csv
data_test.hdf5
label_test.csv
```

The HDF5 files are expected to contain a dataset named:

```text
tracings
```

with samples organized as:

```text
(N, 17, 512)
```

where `N` is the number of EEG windows.

The label files are one-column CSV files containing the corresponding binary class labels.

The split lists under `Data/` provide the exact file assignments used in our experiments. Raw EEG data and the preprocessing pipeline used to construct the HDF5 files are not included in the current release.

---

# Models

## 1. SegFormer + Temporal-Patchify

```text
Models/Segformer_Patchify.py
```

This is the main Temporal-Patchify implementation.

For an EEG input:

```text
(B, 17, 512)
```

a learnable spatial projection first maps the 17 EEG channels to 32 features at each time point:

```text
(B, 17, 512)
        ↓
(B, 32, 512)
```

The resulting representation is flattened and divided into non-overlapping patches of length 32. The patch representation is subsequently arranged into a `512 × 512` image-like representation and replicated across three channels before being passed to a pretrained **SegFormer MiT-B0** backbone.

Conceptually:

```text
17-channel EEG
      ↓
Learnable spatial projection
      ↓
32 × 512 feature map
      ↓
Temporal-Patchify
      ↓
512 × 512 representation
      ↓
3-channel replication
      ↓
SegFormer MiT-B0
      ↓
Binary prediction
```

The backbone used by the implementation is:

```text
nvidia/mit-b0
```

---

## 2. SegFormer Temporal-Tile Representation

```text
Models/Segformer_Temporal_Tile.py
```

This implementation provides the alternative/original SegFormer EEG representation used for comparison with Temporal-Patchify.

The input consists of 17 referential EEG channels, which are converted into 20 longitudinal bipolar derivations before constructing the SegFormer-compatible representation.

This model also uses:

```text
nvidia/mit-b0
```

as the image backbone.

---

## 3. MambaVision Variants

Two MambaVision-based implementations are included:

```text
Models/MambaVision_Input20_512.py
Models/MambaVision_input256_512.py
```

Both models use:

```text
nvidia/MambaVision-T-1K
```

as the pretrained backbone.

### 20-dimensional projection

```text
MambaVision_Input20_512.py
```

maps each time point from 17 EEG channels to 20 learned features:

```text
17 → 20
```

producing an image-like representation of approximately:

```text
20 × 512
```

before replication across three channels.

### 256-dimensional projection

```text
MambaVision_input256_512.py
```

uses a larger learned spatial projection:

```text
17 → 256
```

producing:

```text
256 × 512
```

before being passed to MambaVision.

---

## Ablation Experiments

The following Temporal-Patchify ablation implementations are provided:

```text
Models/Ablation/Segformer_Patchify_A1_L64.py
Models/Ablation/Segformer_Patchify_A2_L128.py
```

These files correspond to alternative Temporal-Patchify configurations evaluated in the ablation experiments.

---

## External Baselines

Implementations of the external comparison models are provided under:

```text
Models/External_Baseline/
```

Currently included:

```text
DSAINet.py
D_RCSAM.py
```

These implementations are adapted to the same EEG input and experimental pipeline to enable comparison under consistent preprocessing and evaluation settings.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/opencode123456/Seizure_Prediction.git
cd Seizure_Prediction
```

A Python environment with the following core packages is required:

```bash
pip install numpy pandas h5py scikit-learn torch transformers accelerate
```

A CUDA-enabled PyTorch installation is recommended for GPU training.

The pretrained SegFormer and MambaVision backbones are loaded through Hugging Face Transformers and therefore need to be available either through an Internet connection or a local Hugging Face cache.

---

# Training

## Prepare the data

For each fold, construct a data directory containing:

```text
data_train.hdf5
label_train.csv
data_test.hdf5
label_test.csv
```

according to the corresponding split definitions:

```text
Data/foldX/train.txt
Data/foldX/test.txt
```

For example:

```text
prepared_data/
└── fold1/
    ├── data_train.hdf5
    ├── label_train.csv
    ├── data_test.hdf5
    └── label_test.csv
```

---

## Train Temporal-Patchify SegFormer

```bash
python Models/Segformer_Patchify.py \
    --data_dir ./prepared_data/fold1 \
    --model_output_dir ./outputs/fold1/Segformer_Patchify \
    --seed 42
```

---

## Train Temporal-Tile SegFormer

```bash
python Models/Segformer_Temporal_Tile.py \
    --data_dir ./prepared_data/fold1 \
    --model_output_dir ./outputs/fold1/Segformer_Temporal_Tile \
    --seed 42
```

---

## Train MambaVision

20-dimensional projection:

```bash
python Models/MambaVision_Input20_512.py \
    --data_dir ./prepared_data/fold1 \
    --model_output_dir ./outputs/fold1/MambaVision20 \
    --seed 42
```

256-dimensional projection:

```bash
python Models/MambaVision_input256_512.py \
    --data_dir ./prepared_data/fold1 \
    --model_output_dir ./outputs/fold1/MambaVision256 \
    --seed 42
```

The same procedure can be repeated for Fold 2–Fold 5.

---

# Training Configuration

The current Temporal-Patchify SegFormer implementation uses the following default configuration:

| Parameter                    |            Value |
| ---------------------------- | ---------------: |
| EEG channels                 |               17 |
| Time points                  |              512 |
| Spatial projection dimension |               32 |
| Patch length                 |               32 |
| Backbone                     | SegFormer MiT-B0 |
| Learning rate                |           `1e-4` |
| Batch size                   |               32 |
| Epochs                       |               10 |
| Weight decay                 |           `0.01` |
| Warm-up ratio                |           `0.01` |

The random seed can be specified through:

```bash
--seed
```

---

# Evaluation

During training, evaluation is performed after each epoch.

The implementations report:

* Accuracy
* Sensitivity
* Specificity
* ROC-AUC

The best checkpoint is selected according to the **highest validation ROC-AUC**.

For five-fold evaluation, the model should be independently trained and evaluated using all five predefined partitions.

---

# Reproducibility

To reproduce the data partition used in our experiments:

1. Obtain TUSZ independently from NEDC.
2. Apply the required EEG preprocessing locally.
3. Match the resulting preprocessed file identifiers to the lists provided under `Data/`.
4. For each fold, construct the training data using `train.txt`.
5. Construct the held-out evaluation data using `test.txt`.
6. Train the desired model independently for each fold.
7. Aggregate the evaluation results across the five folds.

The provided split files are intended to ensure that researchers can reproduce the **same file-level experimental partitions** rather than generating a new random five-fold split.

---

# Data Availability

The original TUSZ EEG recordings and annotations are distributed by the Neural Engineering Data Consortium (NEDC) under their own data-use agreement.

This repository does **not** redistribute:

* raw EEG recordings;
* EDF files;
* original TUSZ annotation files;
* clinical reports; or
* processed EEG signal arrays.

Only the experimental file identifiers and their fold assignments are provided for reproducibility.

Users who wish to reproduce the experiments should obtain the TUSZ corpus independently from NEDC.

---

# Citation

If you use this repository or the provided experimental splits in your research, please cite the corresponding paper.

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

The citation information will be updated after publication.

---

## License

The source code in this repository is provided for academic research and reproducibility purposes.

The TUSZ dataset is **not** covered by this repository and remains subject to the terms and conditions specified by NEDC.
