from .base_dataset import BaseDataset
from .registry import DatasetRegistry
from .cifar10 import CIFAR10Dataset
from .cat_dog import CatDogDataset
from .plate import PlateDataset
from .augmented_plate import AugmentedPlateDataset

__all__ = [
    'BaseDataset',
    'DatasetRegistry',
    'CIFAR10Dataset',
    'CatDogDataset',
    'PlateDataset',
    'AugmentedPlateDataset',
]
