import gzip
from typing import ClassVar, Dict
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import requests

from ml_copypasta.datasets.dataset import Dataset


@dataclass
class CachedData:
    url: str
    path: Path
    parse: callable

    HEADERS: ClassVar[Dict] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __post_init__(self):
        if not self.path.exists():
            print(f"Downloading {self.url} to {self.path}")
            response = requests.get(self.url, headers=self.HEADERS)
            data = gzip.decompress(response.content)
            with self.path.open("wb") as f:
                f.write(data)
        data = self.path.read_bytes()
        self.data = self.parse(data)


@dataclass
class MNIST(Dataset):
    """Blazingly fast MNIST dataset"""

    path: Path = Path.home() / "datasets" / "mnist"

    BASE_URL: ClassVar = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

    def __post_init__(self):
        self.train_dir = self.path / "train"
        self.test_dir = self.path / "test"
        for d in [self.train_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.train_images = CachedData(
            self.BASE_URL + "train-images-idx3-ubyte.gz",
            self.train_dir / "images.gz",
            lambda x: np.frombuffer(x, np.uint8, offset=16).reshape(-1, 28, 28),
        ).data
        self.test_images = CachedData(
            self.BASE_URL + "t10k-images-idx3-ubyte.gz",
            self.test_dir / "images.gz",
            lambda x: np.frombuffer(x, np.uint8, offset=16).reshape(-1, 28, 28),
        ).data
        self.train_labels = CachedData(
            self.BASE_URL + "train-labels-idx1-ubyte.gz",
            self.train_dir / "labels.gz",
            lambda x: np.frombuffer(x, np.uint8, offset=8),
        ).data
        self.test_labels = CachedData(
            self.BASE_URL + "t10k-labels-idx1-ubyte.gz",
            self.test_dir / "labels.gz",
            lambda x: np.frombuffer(x, np.uint8, offset=8),
        ).data


if __name__ == "__main__":
    import time

    st = time.monotonic()

    mnist = MNIST()
    print(mnist.train_images.shape)

    duration = time.monotonic() - st

    print(f"Loaded MNIST in {duration:.2f} seconds")
