# src/dataset.py

import pickle
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "gtsrb"


class GTSRBDataset(Dataset):

    def __init__(self, filename):

        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        print(f"Loading dataset from: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.images = data["features"]
        self.labels = data["labels"]

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.fromarray(self.images[idx])
        image = self.transform(image)

        label = self.labels[idx]

        return image, label