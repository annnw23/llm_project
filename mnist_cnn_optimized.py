import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import optuna
import gradio as gr


# ------- Config -------
N_TRIALS = 20
N_EPOCHS = 3
VAL_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset -------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081)),
])

full_train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

val_n = int(len(full_train_ds)*VAL_RATIO)
train_n = len(full_train_ds) -val_n

train_ds, val_ds = random_split(full_train_ds, [train_n, val_n])


# --------------- CNN ---------------
class CNN(nn.Module):
    def __init__(self, kernel_size: int, conv1_out: int, conv2_out: int, fc_units: int, dropout: float):
        super().__init__()
        pad = kernel_size //2
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_out*7*7, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))



# ----------------  train ----------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)
        loss =criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n = len(loader.dataset)
    return total_loss / n, correct /n



#----- eval ---

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out  = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# --- optuna ---
def objective(trial: optuna.Trial) -> float:
    kernel_size = trial.suggest_categorical("kernel_size", [7])
    conv1_out   = trial.suggest_categorical("conv1_out",   [16, 32])
    conv2_out   = trial.suggest_categorical("conv2_out",   [16, 64, 256])
    fc_units    = trial.suggest_categorical("fc_units",    [256])
    dropout     = trial.suggest_float("dropout", 0.47, 0.47)
    batch_size  = trial.suggest_categorical("batch_size",  [256])
    opt_name    = trial.suggest_categorical("optimizer",   ["Adam", "SGD"])
    lr          = trial.suggest_float("lr", 1e-4, 1e-3, log=True)


    # ---load data ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)


    # ---- optimiziers- ----
    model = CNN(kernel_size, conv1_out, conv2_out, fc_units, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:  # SGD
        momentum = trial.suggest_float("momentum", 0.78, 0.80)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # ------ train and val epochs -------
    print(f"\n  Trial {trial.number:>3} | "
          f"k={kernel_size} c1={conv1_out} c2={conv2_out} fc={fc_units} "
          f"drop={dropout:.2f} bs={batch_size} opt={opt_name} lr={lr:.2e}")
    print(f"  {'Epoch':>5} | {'TrainLoss':>10} {'TrainAcc':>9} | "
          f"{'ValLoss':>9} {'ValAcc':>8}")
    print("  " + "-" * 55)

    val_acc = 0.0
    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        val_acc = vl_acc

        print(f"  {epoch + 1:>5} | {tr_loss:>10.4f} {tr_acc:>9.4f} | "
              f"{vl_loss:>9.4f} {vl_acc:>8.4f}")

        #report intermediate value after each epoch -> need to wait for this testing
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            print("  → Pruned.")
            raise optuna.exceptions.TrialPruned()

    return val_acc


# ----------- run stusy -------------
def main():
    print(f"Device : {DEVICE}")
    print(f"Trials : {N_TRIALS}  |  Epochs/trial : {N_EPOCHS}\n")

    study = optuna.create_study(
        study_name="mnist_cnn_hpo",
        direction="maximize",
        storage="sqlite:///mnist_optuna.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)


# ---------- printing the best fit ---------------------
    best = study.best_trial
    print("\n" + "=" * 55)
    print(f"Best validation accuracy : {best.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:>15} = {v}")


# ------------- wroking with the best params -----------------
    print("\nRetraining best config on full training set …")
    p = best.params
    model = CNN(
        kernel_size=p["kernel_size"],
        conv1_out=p["conv1_out"],
        conv2_out=p["conv2_out"],
        fc_units=p["fc_units"],
        dropout=p["dropout"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    if p["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=p["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=p["lr"], momentum=p.get("momentum", 0.9))

    full_loader = DataLoader(full_train_ds, batch_size=p["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    RETRAIN_EPOCHS = 2
    print(f"  {'Epoch':>5} | {'TrainLoss':>10} {'TrainAcc':>9}")
    print("  " + "-" * 30)
    for epoch in range(1, RETRAIN_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, full_loader, optimizer, criterion)
        print(f"  {epoch:>5} | {tr_loss:>10.4f} {tr_acc:>9.4f}")

    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"\n  Test loss     : {te_loss:.4f}")
    print(f"  Test accuracy : {te_acc:.4f}  ({te_acc * 100:.2f}%)")

    torch.save(model.state_dict(), "mnist_cnn_best.pth")
    print("\nSaved → mnist_cnn_best.pth")
    print("Optuna DB → mnist_optuna.db  (view with: optuna-dashboard sqlite:///mnist_optuna.db)")


if __name__ == "__main__":
    main()
