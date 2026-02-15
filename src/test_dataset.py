# src/test_dataset.py

from torch.utils.data import DataLoader
from src.dataset import GTSRBDataset


def main():

    dataset = GTSRBDataset("train.p")

    print("Dataset size:", len(dataset))

    image, label = dataset[0]

    print("Image shape:", image.shape)
    print("Label:", label)

    loader = DataLoader(dataset, batch_size=32)

    images, labels = next(iter(loader))

    print("Batch shape:", images.shape)


if __name__ == "__main__":
    main()