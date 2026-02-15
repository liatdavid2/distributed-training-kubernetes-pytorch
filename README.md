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
