import argparse
from getpass import getpass

import keyring
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from comet_ml import Experiment
from tqdm import tqdm

from config import CIFAR_100_MEAN, CIFAR_100_STD, PROJECT
from nets.densenet import DenseNet


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
        default=True,
        help="Whether to log progress to TensorBoard",
        action="store_true",
    )
    return parser


def train_and_evaluate():
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

    args = read_args().parse_args()
    args.growth_rate = 32
    args.block_config = (6, 12, 24, 16)
    args.num_init_features = 64
    args.bn_size = 4
    args.num_classes = 1000

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
        experiment.log_parameters(vars(args))

    model = DenseNet(
        growth_rate=args.growth_rate,
        block_config=args.block_config,
        num_init_features=args.num_init_features,
        bn_size=args.bn_size,
        drop_rate=args.dropout,
        num_classes=args.num_classes,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with experiment.train():
        model.train()
        step = 0
        for epoch in range(args.num_epochs):
            experiment.log_current_epoch(epoch)
            correct = 0
            total = 0
            with tqdm(
                total=len(train_data), desc="Epoch {}: Step".format(epoch)
            ) as pbar:
                for idx, (images, labels) in enumerate(train_loader):
                    images = Variable(images.cuda())
                    labels = Variable(images.cuda())

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    experiment.log_metric("loss", loss.data.item())

                    total += labels.size(0)
                    correct += float((outputs.data == labels.data).sum())
                    experiment.log_metric(
                        "accuracy", 100 * correct / total, step=step
                    )
                    step += 1
                    if (idx + 1) % 100 == 0:
                        print(
                            "Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
                            % (
                                epoch + 1,
                                args.num_epochs,
                                idx + 1,
                                len(train_data) // args.batch_size,
                                loss.data.item(),
                            )
                        )
                    pbar.update()

        with experiment.test():
            model.eval()

            correct = 0
            total = 0
            with tqdm(
                total=len(test_data), desc="Test Step".format(epoch)
            ) as pbar:
                for idx, (images, labels) in enumerate(test_loader):
                    images = Variable(images.cuda())
                    labels = Variable(images.cuda())

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total += labels.size(0)
                    correct += float((outputs.data == labels.data).sum())
            accuracy = 100 * correct / total
            experiment.log_metric("accuracy", accuracy)
            print(
                "Test accuracy on {} images: {}".format(
                    len(test_data), accuracy
                )
            )

