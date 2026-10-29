"""
Fashion-MNIST clothing classifier prototype.

A customer request: an online clothing retailer wants to explore
automatically assigning broad product categories to small grayscale
catalog images. This prototype classifies a 28 by 28 grayscale image
into one of 10 clothing categories, shows how the network learns,
reports performance on images it did not see during training, and
tests how one deliberate change affects the result.

This is a prototype and experiment, not a production service.

Dataset: Fashion-MNIST, loaded through Torchvision.
    Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST:
    A Novel Image Dataset for Benchmarking Machine Learning Algorithms.
    arXiv:1708.07747. The dataset is distributed under the MIT License.

How to run:
    python -m pip install -r requirements.txt
    python fashion_mnist.py

The script prints progress as it goes and writes three files:
    loss_curves.png   training and validation loss for the baseline run
    results.csv       epoch-by-epoch losses and validation accuracy
"""

import csv
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Fixed seed so the split and the starting weights are repeatable.
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

BATCH_SIZE = 64
EPOCHS = 5
BASELINE_LR = 0.001
EXPERIMENT_LR = 0.01

# The 10 Fashion-MNIST categories, in the order Torchvision uses.
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data():
    """Load Fashion-MNIST and split the training set.

    Returns train, validation, and test loaders. The test set is kept
    untouched until the final evaluation at the end of the script.
    """
    transform = transforms.ToTensor()

    full_train = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # 50,000 training images and 10,000 validation images, split with
    # the fixed seed so the same images land in each set every run.
    generator = torch.Generator().manual_seed(SEED)
    train_data, val_data = random_split(
        full_train, [50000, 10000], generator=generator
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


class ClothingNet(nn.Module):
    """A simple multi-layer perceptron for 28 by 28 grayscale images.

    Each image is flattened into 784 numbers (28 times 28 pixels).
    One hidden layer of 128 units learns combinations of pixels, and
    the output layer produces one score for each of the 10 categories.

    Activation: ReLU. ReLU keeps positive values and turns negative
    values to zero. It is a good default for a hidden layer because it
    lets the network learn non-linear patterns while keeping the
    gradients healthy, unlike a sigmoid, which can squash gradients
    toward zero in deeper networks. The output layer has no activation
    because CrossEntropyLoss expects raw scores (logits).
    """

    def __init__(self, hidden_size=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28 * 28, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x


def evaluate(model, loader, loss_fn):
    """Return the average loss and accuracy of a model on a loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            total_loss += loss_fn(scores, labels).item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train_run(train_loader, val_loader, learning_rate, run_name):
    """Train one model and record training and validation loss per epoch.

    One epoch is one full pass through the training images. For each
    batch the loop does a forward pass, measures the loss, computes
    gradients with backward, and updates the weights with the optimizer.
    """
    model = ClothingNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            scores = model(images)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn)
        history.append((epoch, train_loss, val_loss, val_accuracy))
        print(
            f"{run_name} epoch {epoch}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, "
            f"val accuracy {val_accuracy:.4f}"
        )
    return model, history


def main():
    print("Loading Fashion-MNIST...")
    train_loader, val_loader, test_loader = load_data()
    print(
        f"train images: {len(train_loader.dataset)}, "
        f"validation images: {len(val_loader.dataset)}, "
        f"test images: {len(test_loader.dataset)}"
    )

    loss_fn = nn.CrossEntropyLoss()

    # ---- Baseline run: Adam learning rate 0.001 ----
    print("\nBaseline run: learning rate 0.001")
    baseline_model, baseline_history = train_run(
        train_loader, val_loader, BASELINE_LR, "baseline"
    )

    # ---- Prediction for the experiment, written BEFORE running it ----
    # Raising the learning rate ten times, from 0.001 to 0.01, should
    # make the model learn faster at first, because each weight update
    # takes a bigger step toward lower loss. But bigger steps can also
    # overshoot the best weights, so I predict the validation loss will
    # bounce around more, end up higher than the baseline, and the final
    # validation accuracy will be similar or a little lower than the
    # baseline. In short: faster early progress, less stable training,
    # no better final result.
    print("\nPrediction before the experiment run:")
    print(
        "With learning rate 0.01 the model should learn faster at "
        "first, but the bigger steps may overshoot, so validation "
        "loss should bounce more and finish higher than the baseline, "
        "with similar or slightly lower validation accuracy."
    )

    # ---- Experiment run: only the learning rate changes ----
    print("\nExperiment run: learning rate 0.01 (everything else identical)")
    experiment_model, experiment_history = train_run(
        train_loader, val_loader, EXPERIMENT_LR, "experiment"
    )

    # ---- Final evaluation on the untouched test set ----
    # The test set played no part in choosing the architecture, the
    # learning rate, or the number of epochs. This is its first use.
    baseline_test_loss, baseline_test_accuracy = evaluate(
        baseline_model, test_loader, loss_fn
    )
    experiment_test_loss, experiment_test_accuracy = evaluate(
        experiment_model, test_loader, loss_fn
    )
    print(f"\nFinal test accuracy (baseline, lr 0.001): "
          f"{baseline_test_accuracy:.4f}")
    print(f"Final test accuracy (experiment, lr 0.01): "
          f"{experiment_test_accuracy:.4f}")

    # ---- Save the epoch-by-epoch results ----
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "epoch", "train_loss", "val_loss",
                         "val_accuracy"])
        for epoch, tl, vl, va in baseline_history:
            writer.writerow(["baseline", epoch, f"{tl:.4f}",
                             f"{vl:.4f}", f"{va:.4f}"])
        for epoch, tl, vl, va in experiment_history:
            writer.writerow(["experiment", epoch, f"{tl:.4f}",
                             f"{vl:.4f}", f"{va:.4f}"])
        writer.writerow(["baseline", "test", "", "",
                         f"{baseline_test_accuracy:.4f}"])
        writer.writerow(["experiment", "test", "", "",
                         f"{experiment_test_accuracy:.4f}"])
    print("Wrote results.csv")

    # ---- Plot the baseline loss curves ----
    epochs = [h[0] for h in baseline_history]
    plt.figure()
    plt.plot(epochs, [h[1] for h in baseline_history], label="train loss")
    plt.plot(epochs, [h[2] for h in baseline_history], label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.savefig("loss_curves.png")
    print("Wrote loss_curves.png")


if __name__ == "__main__":
    main()
