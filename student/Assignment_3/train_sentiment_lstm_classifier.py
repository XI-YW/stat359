#!/usr/bin/env python
# coding: utf-8

import os
import re
import random
import numpy as np
import pandas as pd
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import gensim.downloader as api
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# =========================
# Reproducibility + Device
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Tokenization + FastText sequence builder
# =========================
_token_re = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str):
    return _token_re.findall(text.lower())


def safe_fasttext_vector(ft, token: str):
    """
    For fasttext-wiki-news-subwords-300, most tokens exist.
    Handle any edge cases gracefully.
    """
    # Newer gensim KeyedVectors has get_vector
    if hasattr(ft, "get_vector"):
        try:
            return ft.get_vector(token)
        except KeyError:
            return None
    else:
        try:
            return ft[token]
        except KeyError:
            return None


def sentence_to_padded_matrix(sentence: str, ft, max_len: int = 32, dim: int = 300) -> np.ndarray:
    """
    Returns (max_len, dim) float32 matrix:
    - tokenize sentence
    - map tokens to fastText vectors
    - truncate/pad to exactly max_len
    """
    toks = tokenize(sentence)
    mat = np.zeros((max_len, dim), dtype=np.float32)

    # Fill sequentially until max_len
    j = 0
    for tok in toks:
        if j >= max_len:
            break
        vec = safe_fasttext_vector(ft, tok)
        if vec is None:
            continue
        mat[j] = vec.astype(np.float32, copy=False)
        j += 1

    return mat


def build_sequence_tensor(sentences, ft, max_len: int = 32, dim: int = 300) -> np.ndarray:
    X = np.zeros((len(sentences), max_len, dim), dtype=np.float32)
    for i, s in enumerate(tqdm(sentences, desc="Building FastText sequences")):
        X[i] = sentence_to_padded_matrix(s, ft, max_len=max_len, dim=dim)
    return X


# =========================
# Dataset
# =========================
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# LSTM Model
# =========================
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, 32, 300)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # last layer forward is at -2, last layer backward is at -1
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim*2)
        else:
            h = h_n[-1]  # (batch, hidden_dim)

        return self.fc(h)


# =========================
# Train/Eval loops
# =========================
@torch.no_grad()
def eval_loop(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, np.array(all_labels), np.array(all_preds)


def train_loop(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for xb, yb in tqdm(loader, desc="Training", leave=False):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(yb.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def save_three_panel_learning_curves(
    train_loss, val_loss,
    train_f1, val_f1,
    train_acc, val_acc,
    out_path: str,
):
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(12, 15))

    # Loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Macro F1
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_f1, label="Train F1")
    plt.plot(epochs, val_f1, label="Val F1")
    plt.title("F1 Macro Score Curve")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main():
    set_seed(42)
    device = get_device()
    print(f"\nUsing device: {device}")

    os.makedirs("outputs", exist_ok=True)

    print("\n========== Loading Dataset ==========")
    dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    data = pd.DataFrame(dataset["train"])
    print("Dataset loaded. Example:", dataset["train"][0])
    print(f"DataFrame shape: {data.shape}")

    # quick length stats (optional but nice)
    sentence_lengths = data["sentence"].apply(lambda x: len(x.split()))
    print("\nSentence length statistics:")
    print(sentence_lengths.describe())
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=30, edgecolor="black")
    plt.title("Distribution of Sentence Lengths")
    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/lstm_sentence_length_hist.png", dpi=200)
    plt.close()
    print("Saved sentence length histogram: outputs/lstm_sentence_length_hist.png")

    texts = data["sentence"].tolist()
    y = data["label"].values.astype(np.int64)

    print("\n========== Splitting Data (Stratified) ==========")
    idx = np.arange(len(texts))

    idx_trainval, idx_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15, stratify=y_trainval, random_state=42)

    def gather(indices):
        return [texts[i] for i in indices], y[indices]

    train_texts, y_train = gather(idx_train)
    val_texts, y_val = gather(idx_val)
    test_texts, y_test = gather(idx_test)

    print(f"Sizes: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    print("\n========== Loading FastText (gensim) ==========")
    print("Downloading/loading: fasttext-wiki-news-subwords-300 (first run may take a while).")
    ft = api.load("fasttext-wiki-news-subwords-300")

    print("\n========== Building padded FastText sequences ==========")
    max_len = 32
    dim = 300
    X_train = build_sequence_tensor(train_texts, ft, max_len=max_len, dim=dim)
    X_val   = build_sequence_tensor(val_texts, ft, max_len=max_len, dim=dim)
    X_test  = build_sequence_tensor(test_texts, ft, max_len=max_len, dim=dim)

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X_test, y_test), batch_size=32, shuffle=False)

    print("\n========== Defining LSTM Model ==========")
    num_classes = 3
    model = LSTMClassifier(
        input_dim=dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        bidirectional=True,
    ).to(device)

    # class weights computed from TRAIN split (don’t hardcode)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = (len(y_train) / (num_classes * np.maximum(counts, 1.0))).astype(np.float32)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print("Class counts (train):", counts)
    print("Class weights used:", weights)

    print("\n========== Starting Training Loop ==========")
    num_epochs = 30
    best_val_f1 = 0.0

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    train_f1_hist, val_f1_hist = [], []

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        tr_loss, tr_acc, tr_f1 = train_loop(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, va_f1, _, _ = eval_loop(model, val_loader, device, criterion)

        train_loss_hist.append(tr_loss); val_loss_hist.append(va_loss)
        train_acc_hist.append(tr_acc);   val_acc_hist.append(va_acc)
        train_f1_hist.append(tr_f1);     val_f1_hist.append(va_f1)

        scheduler.step(va_f1)

        print(
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Train Macro-F1: {tr_f1:.4f}\n"
            f"Val   Loss: {va_loss:.4f}, Val   Acc: {va_acc:.4f}, Val   Macro-F1: {va_f1:.4f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), "outputs/best_lstm_model.pth")
            print(f">>> Saved new best model (Val Macro-F1: {best_val_f1:.4f})")

    print("\n========== Saving Learning Curves (3-panel) ==========")
    save_three_panel_learning_curves(
        train_loss_hist, val_loss_hist,
        train_f1_hist, val_f1_hist,
        train_acc_hist, val_acc_hist,
        out_path="outputs/lstm_learning_curves.png",
    )
    print("Saved: outputs/lstm_learning_curves.png")

    print("\n========== Evaluating on Test Set ==========")
    model.load_state_dict(torch.load("outputs/best_lstm_model.pth", map_location=device))
    test_loss, test_acc, test_f1, y_true, y_pred = eval_loop(model, test_loader, device, criterion)

    print("\n" + "=" * 50)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Macro-F1: {test_f1:.4f}")
    print("=" * 50 + "\n")

    class_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plot_confusion_matrix(cm, class_names, "outputs/lstm_confusion_matrix.png")
    print("Saved confusion matrix: outputs/lstm_confusion_matrix.png")

    with open("outputs/lstm_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"best_val_macro_f1\t{best_val_f1:.6f}\n")
        f.write(f"test_loss\t{test_loss:.6f}\n")
        f.write(f"test_accuracy\t{test_acc:.6f}\n")
        f.write(f"test_macro_f1\t{test_f1:.6f}\n")
    print("Saved metrics: outputs/lstm_metrics.txt")

    if test_f1 < 0.70:
        print("\nWARNING: Test Macro-F1 < 0.70.")
        print("Try: hidden_dim=256, dropout=0.2, lr=3e-4, or keep bidirectional=True.")

    print("\n========== Script Complete ==========")


if __name__ == "__main__":
    main()
