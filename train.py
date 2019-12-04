import argparse
from getpass import getpass

import keyring
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from comet_ml import Experiment
from tqdm import tqdm

from config import CIFAR_100_MEAN, CIFAR_100_STD, PROJECT


def read_args():
    parser = argparse.ArgumentParser(
        description="Image classification demo "
        + "of DenseNet in PyTorch on CIFAR 100"
    )

    parser.add_argument(
        "--num-epochs",
        default=100,
        type=int,
        help="Number of total epochs to run",
    )

    parser.add_argument(
        "--batch-size", default=64, type=int, help="Batch size (default: 64)"
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-2,
        type=float,
        help="Initial learning rate (default: 1e-2)",
    )

    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        help="Weight decay (default: 1e-4)",
    )

    parser.add_argument(
        "--dropout", default=0, type=float, help="Dropout rate (default: 0.0)"
    )

    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
        help="Path to the latest checkpoint (default: none)",
    )

    parser.add_argument(
        "--comet-logging",
        help="Whether to log progress to TensorBoard",
        action="store_true",
    )
    return parser


def train_and_evaluate():
    args = read_args().parse_args()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=np.asarray(CIFAR_100_MEAN) / 255.0,
                std=np.asarray(CIFAR_100_STD) / 255.0,
            ),
        ]
    )
    data_dirname = "./data"

    train_data = datasets.CIFAR100(
        root=data_dirname, train=True, download=True, transform=transform
    )
    test_data = datasets.CIFAR100(
        root=data_dirname, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    if args.comet_logging:
        if not keyring.get_password("comet", PROJECT):
            comet_ml_api_key = getpass("Please enter the comet.ml API key: ")
            keyring.set_password("comet", PROJECT, comet_ml_api_key)
        else:
            comet_ml_api_key = keyring.get_password("comet", PROJECT)

        experiment = Experiment(
            api_key=comet_ml_api_key,
            project_name=PROJECT,
            workspace=PROJECT,
            auto_output_logging=None,
        )

