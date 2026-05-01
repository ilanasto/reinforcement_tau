"""
HW1 - part 3: deeper model on MNIST.

Extends the linear model from mnist_2.py with a hidden layer:
    784 -> Linear -> 500 -> ReLU -> Linear -> 10

Uses the best optimizer configuration found in part 2 (Adam + normalization).
Trains for NUM_EPOCHS, evaluates on the test set once at the end, and saves
the final model plus the training-loss curve.
"""

import os
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as T
import matplotlib.pyplot as plt


DATA_DIR = "./data"
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

INPUT_SIZE = 784
HIDDEN_SIZE = 500
NUM_CLASSES = 10

NUM_EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3


def mnist_stats():
    """Compute MNIST train pixel mean/std from the data itself."""
    ds = dsets.MNIST(DATA_DIR, train=True, transform=T.ToTensor(), download=True)
    x = torch.stack([img for img, _ in ds])
    return x.mean().item(), x.std().item()


MEAN, STD = mnist_stats()
print(f"MNIST train stats: mean={MEAN:.4f}, std={STD:.4f}")


def loaders(batch_size):
    """Load MNIST with normalization and return (train, test) loaders."""
    tfm = T.Compose([T.ToTensor(), T.Normalize((MEAN,), (STD,))])
    train = dsets.MNIST(DATA_DIR, train=True, transform=tfm, download=True)
    test = dsets.MNIST(DATA_DIR, train=False, transform=tfm, download=True)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=512, shuffle=False),
    )


class Net(nn.Module):
    """784 -> 500 -> ReLU -> 10"""

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def evaluate(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (net(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train():
    torch.manual_seed(0)
    train_loader, test_loader = loaders(BATCH_SIZE)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n=== deeper MLP (784 -> {HIDDEN_SIZE} -> 10), {n_params:,} params ===")
    print(f"    epochs={NUM_EPOCHS}, bs={BATCH_SIZE}, optimizer=Adam(lr={LR}), normalize=True")

    losses = []
    for epoch in range(NUM_EPOCHS):
        net.train()
        epoch_loss = 0.0
        n = 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * y.size(0)
            n += y.size(0)
        losses.append(epoch_loss / n)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{NUM_EPOCHS}  loss={losses[-1]:.4f}")

    acc = evaluate(net, test_loader)
    path = os.path.join(MODELS_DIR, "mnist_3_deeper.pkl")
    torch.save(net.state_dict(), path)
    print(f"  final test acc = {acc*100:.2f}%   saved to {path}")
    return losses, acc


def main():
    losses, acc = train()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), losses, label="deeper MLP (784-500-10)")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.yscale("log")
    plt.title(f"MNIST training loss - deeper model ({NUM_EPOCHS} epochs)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join('./', "loss_curve_mnist_3.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nLoss curve saved to {plot_path}")
    print(f"\n--- summary ---")
    print(f"deeper MLP 784-{HIDDEN_SIZE}-10   test_acc={acc*100:.2f}%")


if __name__ == "__main__":
    main()
