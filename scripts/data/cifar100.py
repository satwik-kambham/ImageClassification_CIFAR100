from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset_builder, load_dataset
import albumentations as A
import lightning.pytorch as pl


def load_cifar100():
    ds_builder = load_dataset_builder("cifar100")
    features = ds_builder.info.features
    train_ds = load_dataset("cifar100", split="train")
    train_ds.set_format("torch")
    test_ds = load_dataset("cifar100", split="test")
    test_ds.set_format("torch")

    img_tfms = transforms.Compose(
        [
            transforms.Normalize(
                (129.37731888, 124.10583864, 112.47758569),
                (51.24804743, 50.64248745, 51.61805643),
            )
        ]
    )

    aug = A.Compose(
        [
            A.RandomRotate90(),
        ]
    )

    return train_ds, test_ds, features, img_tfms, aug


def apply_augmentations_and_transforms(img, aug, tfms):
    return aug(image=tfms(img.permute(2, 1, 0).float()).permute(2, 1, 0).numpy())[
        "image"
    ]


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        (
            self.train_ds,
            self.test_ds,
            self.features,
            self.img_tfms,
            self.aug,
        ) = load_cifar100()

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
