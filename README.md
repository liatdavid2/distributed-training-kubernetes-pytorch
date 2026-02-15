# Kubernetes PyTorch Distributed Training – GTSRB

This project implements a PyTorch data pipeline and distributed training foundation for image classification using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
It demonstrates how to load data, convert it into tensors, and prepare it for scalable distributed training using Kubernetes and PyTorch Distributed Data Parallel (DDP).

The project is designed as a production-style ML training system, supporting local training, Docker containers, and multi-node Kubernetes execution.

---

# Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The dataset contains:

* 34,799 training images
* 43 traffic sign classes
* RGB images stored as numpy arrays inside pickle files

Dataset files:

```
data/gtsrb/train.p
data/gtsrb/valid.p
data/gtsrb/test.p
```

Each `.p` file contains:

```
{
    "features": numpy array of shape (N, H, W, 3)
    "labels": numpy array of shape (N,)
}
```

Where:

* features → image pixel data
* labels → class index (0–42)

---

# Project Structure

```
kubernetes-pytorch-distributed-training/
│
├── src/
│   ├── dataset.py
│   └── test_dataset.py
│
├── data/
│   └── gtsrb/
│       ├── train.p
│       ├── valid.p
│       ├── test.p
│       └── signnames.csv
│
├── docker/
│
├── k8s/
│
├── requirements.txt
└── README.md
```

---

# Setup

Clone the repository:

```
git clone https://github.com/liatdavid2/kubernetes-pytorch-distributed-training.git
cd kubernetes-pytorch-distributed-training
```

Create virtual environment:

Windows (Git Bash):

```
py -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Stage 1 – Dataset Loading Pipeline

This stage verifies that the dataset can be loaded and prepared for training.

Pipeline flow:

```
train.p
   ↓
Dataset
   ↓
DataLoader
   ↓
Batch tensors
   ↓
Ready for training
```

What happens:

* The `train.p` file is loaded from disk
* `GTSRBDataset` provides indexed access to images and labels
* Images are converted to PyTorch tensors
* `DataLoader` groups samples into batches

Run:

```
python -m src.test_dataset
```

Expected output:

```
Loading dataset from: data/gtsrb/train.p
Dataset size: 34799
Batch shape: torch.Size([32, 3, 64, 64])
```

This confirms the data pipeline works and is ready for model training.

---

# Stage 2 – Dataset Loader Validation

## Overview

This stage validates that the dataset loading pipeline is correctly implemented and ready for training.

The goal is to ensure that:

* The dataset file can be loaded successfully
* Images and labels are returned in the correct format
* The PyTorch `Dataset` and `DataLoader` work properly
* Batch loading behaves as expected
* A sample batch artifact is saved for reproducibility and debugging

This validation step is critical before starting model training or distributed training.

---

## Script

```text
src/validate_dataset_loader.py
```

---

## How to run

From the project root directory:

```bash
python -m src.validate_dataset_loader
```

On Windows CMD:

```cmd
cd C:\Users\liat\Documents\work\distributed-training-kubernetes-pytorch

python -m src.validate_dataset_loader
```

---

## What the script does

The script performs the following validation steps:

### 1. Load dataset file

Loads the dataset from:

```text
data/gtsrb/train.p
```

Verifies that the file exists and can be parsed correctly.

---

### 2. Initialize PyTorch Dataset

Creates a Dataset instance and verifies:

* Dataset size
* Image shape
* Label format

Expected format:

```text
Image shape: [3, 64, 64]
Label: integer class index
```

---

### 3. Initialize DataLoader

Creates a PyTorch DataLoader and loads a batch.

Validates batch structure:

```text
Images shape: [batch_size, 3, 64, 64]
Labels shape: [batch_size]
```

Example:

```text
Batch images shape: torch.Size([32, 3, 64, 64])
Batch labels shape: torch.Size([32])
```

---

### 4. Save sample batch artifact

Saves a sample batch to:

```text
artifacts/dataset/sample_batch.pt
```

This artifact is used for:

* Debugging
* Model validation
* Pipeline reproducibility
* Distributed training validation

---

## Example output

```text
=== Stage 2: Dataset Loader Test ===

Loading dataset from: data/gtsrb/train.p
Dataset size: 34799

Single image shape: torch.Size([3, 64, 64])
Single label: 41

Batch images shape: torch.Size([32, 3, 64, 64])
Batch labels shape: torch.Size([32])

Saved sample batch artifact to:
artifacts/dataset/sample_batch.pt

Stage 2 completed successfully.
```

---

## Output artifact

Created file:

```text
artifacts/dataset/sample_batch.pt
```

This file contains:

```python
{
    "images": Tensor[batch_size, 3, 64, 64],
    "labels": Tensor[batch_size]
}
```

---

## Why this stage is important

This validation ensures that:

* The dataset pipeline works correctly
* The model can safely consume the data
* Training will not fail due to data format issues
* The pipeline is ready for distributed training (multi-GPU / Kubernetes)

This stage is a standard production practice in ML pipelines to ensure data integrity before training.

