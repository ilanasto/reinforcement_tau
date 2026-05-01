"""
HW1 - part 2: better optimization configurations on MNIST.

Same model as mnist.py
Only the optimizer / batch size/ normalization / epochs change. 
Trains each config for 100 epochs, plots all training-loss curves on one figure.
Prints final test accuracy and the saved model path for each.
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

NUM_EPOCHS = 100


def mnist_stats():
    """
    Compute MNIST train pixel mean/std from the data itself.
    This is used to normalize the training data.
    """
    ds = dsets.MNIST(DATA_DIR, train=True, transform=T.ToTensor(), download=True)
    x = torch.stack([img for img, _ in ds])
    return x.mean().item(), x.std().item()


MEAN, STD = mnist_stats()
print(f"MNIST train stats: mean={MEAN:.4f}, std={STD:.4f}")


def loaders(batch_size, normalize):
    """
    Load the MNIST dataset and return the train and test loaders.
    If normalize is True, the data is normalized to have mean 0 and std 1.
    """
    tfm = [T.ToTensor()]
    if normalize:
        tfm.append(T.Normalize((MEAN,), (STD,)))
    tfm = T.Compose(tfm)
    train = dsets.MNIST(DATA_DIR, train=True, transform=tfm, download=True)
    test = dsets.MNIST(DATA_DIR, train=False, transform=tfm, download=True)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=512, shuffle=False),
    )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc1(x.view(-1, 784))


def evaluate(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (net(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train(name, optimizer_fn, batch_size=100, normalize=False):
    torch.manual_seed(0)
    train_loader, test_loader = loaders(batch_size, normalize)
    net = Net()
    opt = optimizer_fn(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    print(f"\n=== {name} ===")
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
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}/{NUM_EPOCHS}  loss={losses[-1]:.4f}")

    acc = evaluate(net, test_loader)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    torch.save(net.state_dict(), path)
    print(f"  final test acc = {acc*100:.2f}%   saved to {path}")
    return losses, acc, path

"""
Configurations to train the model.
Changing: learning rates, mini-batch size, no. of epochs, normalization, optimizer.
The new configuration should result in faster convergence.
"""
CONFIGS = [
    ## baseline
    ("a_baseline_sgd_1e-3", lambda p: torch.optim.SGD(p, lr=1e-3), 100, False),
    ## changing learning rate
    ("b_sgd_lr_0.1", lambda p: torch.optim.SGD(p, lr=0.1), 100, False),
    ## changing momentum
    ("c_sgd_momentum", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9), 100, False), # momentum=0.9 is a good choice for SGD
    ## changing batch size
    ("d_sgd_momentum_normalize", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),  128, True),
    ## changing normalization
    ("e_sgd_momentum_normalize", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),  128, True),
    ## changing optimizer
    ("f_adam", lambda p: torch.optim.Adam(p, lr=1e-3), 128, True),
]


def main():
    results = []
    for name, opt_fn, bs, norm in CONFIGS:
        losses, acc, path = train(name, opt_fn, bs, norm)
        results.append((name, losses, acc, path))

    plt.figure(figsize=(10, 6))
    for name, losses, _, _ in results:
        plt.plot(range(1, NUM_EPOCHS + 1), losses, label=name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.yscale("log")
    plt.title(f"MNIST training loss ({NUM_EPOCHS} epochs)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join('./', "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nLoss curves saved to {plot_path}")

    print("\n--- summary ---")
    for name, _, acc, path in results:
        print(f"{name:30s} acc={acc*100:5.2f}%   model={path}")


if __name__ == "__main__":
    main()
