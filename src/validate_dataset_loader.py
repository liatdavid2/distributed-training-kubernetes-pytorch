# src/test_dataset.py

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset import GTSRBDataset


# =========================
# Artifacts directory
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "dataset"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_PATH = ARTIFACTS_DIR / "sample_batch.pt"


# =========================
# Main test function
# =========================

def main():

    print("\n=== Stage 2: Dataset Loader Test ===\n")

    # -------------------------
    # Load dataset
    # -------------------------

    dataset = GTSRBDataset("train.p")

    print(f"Dataset size: {len(dataset)}")

    # -------------------------
    # Test single sample
    # -------------------------

    image, label = dataset[0]

    print(f"Single image shape: {image.shape}")
    print(f"Single label: {label}")

    # -------------------------
    # Create DataLoader
    # -------------------------

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    # -------------------------
    # Load one batch
    # -------------------------

    images, labels = next(iter(loader))

    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")

    # -------------------------
    # Save artifact checkpoint
    # -------------------------

    torch.save(
        {
            "images": images,
            "labels": labels
        },
        ARTIFACT_PATH
    )

    print(f"\nSaved sample batch artifact to:")
    print(ARTIFACT_PATH)

    print("\nStage 2 completed successfully.\n")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    main()