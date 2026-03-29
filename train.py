import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dataset import WLASLBodyPartDataset
from model   import MultiStreamSLR


# ──────────────────────────────────────────────
#  Label-smoothing cross-entropy
# ──────────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_labels * log_probs).sum(dim=-1).mean()


# ──────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────

def move_batch(batch, device):
    face, lh, rh, labels = batch
    return (
        face.to(device),
        lh.to(device),
        rh.to(device),
        labels.to(device),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        face, lh, rh, labels = move_batch(batch, device)
        logits = model(face, lh, rh)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# ──────────────────────────────────────────────
#  Training loop (one epoch)
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for step, batch in enumerate(loader, 1):
        face, lh, rh, labels = move_batch(batch, device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(face, lh, rh)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if step % 20 == 0:
            print(f"  step {step:4d}/{len(loader)} | "
                  f"loss {total_loss/total:.4f} | "
                  f"acc {correct/total:.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    return total_loss / total, correct / total


# ──────────────────────────────────────────────
#  Main training function
# ──────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Device : {device}")
    print(f"  Config : {cfg}")
    print(f"{'='*55}\n")

    # ── datasets & loaders ────────────────────────────────────────────
    train_ds = WLASLBodyPartDataset(
        root       = cfg["data_root"],
        split      = "train",
        num_frames = cfg["num_frames"],
        img_size   = cfg["img_size"],
    )
    val_ds = WLASLBodyPartDataset(
        root       = cfg["data_root"],
        split      = "val",
        num_frames = cfg["num_frames"],
        img_size   = cfg["img_size"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = cfg["num_workers"],
        pin_memory  = True,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = MultiStreamSLR(
        num_classes     = train_ds.num_classes,
        feat_dim        = cfg["feat_dim"],
        num_frames      = cfg["num_frames"],
        dropout         = cfg["dropout"],
        pretrained      = True,
        freeze_backbone = True,          # stage-1: freeze CNN
    ).to(device)

    criterion = LabelSmoothingCE(train_ds.num_classes, smoothing=0.1)
    scaler    = torch.amp.GradScaler()

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    # ══════════════════════════════════════════
    #  STAGE 1 — freeze backbone, train heads
    # ══════════════════════════════════════════
    print("\n[Stage 1] Training heads only (backbone frozen)\n")

    trainable, frozen = count_params(model)
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {frozen:,}\n")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["lr_stage1"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr        = cfg["lr_stage1"],
        epochs        = cfg["epochs_stage1"],
        steps_per_epoch=len(train_loader),
        pct_start     = 0.3,
    )

    for epoch in range(1, cfg["epochs_stage1"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{cfg['epochs_stage1']} "
              f"({elapsed:.0f}s) | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print(f"  ✓ Saved best model (val acc = {best_acc:.4f})")

    # ══════════════════════════════════════════
    #  STAGE 2 — unfreeze backbone, fine-tune
    # ══════════════════════════════════════════
    print("\n[Stage 2] Full fine-tuning (backbone unfrozen)\n")

    model.unfreeze_backbones()

    trainable, _ = count_params(model)
    print(f"  Trainable params : {trainable:,}\n")

    face_cnn_params = list(model.face_encoder.features.parameters()) + \
                      list(model.face_encoder.avgpool.parameters())

    optimizer = AdamW(
        [
            {"params": model.hand_encoder.backbone.parameters(), "lr": cfg["lr_backbone"]},
            {"params": face_cnn_params,                          "lr": cfg["lr_backbone"]},
            {"params": model.hand_encoder.proj.parameters(),     "lr": cfg["lr_stage2"]},
            {"params": model.face_encoder.proj.parameters(),     "lr": cfg["lr_stage2"]},
            {"params": model.temporal.parameters(),              "lr": cfg["lr_stage2"]},
            {"params": model.classifier.parameters(),            "lr": cfg["lr_stage2"]},
        ],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr        = [cfg["lr_backbone"]] * 2 + [cfg["lr_stage2"]] * 4,
        epochs        = cfg["epochs_stage2"],
        steps_per_epoch=len(train_loader),
        pct_start     = 0.2,
    )

    for epoch in range(1, cfg["epochs_stage2"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{cfg['epochs_stage2']} "
              f"({elapsed:.0f}s) | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print(f"  ✓ Saved best model (val acc = {best_acc:.4f})")

    print(f"\n{'='*55}")
    print(f"  Training complete | Best val acc: {best_acc:.4f}")
    print(f"  Checkpoint saved to: {ckpt_dir / 'best_model.pth'}")
    print(f"{'='*55}")


# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

CONFIG = {
    # ── paths ────────────────────────────────
    "data_root" : "/kaggle/input/datasets/nguyenthanhvung/cropdataset/crops",
    "ckpt_dir"  : "/kaggle/working/checkpoints",

    # ── data ─────────────────────────────────
    "num_frames"  : 16,
    "img_size"    : 112,
    "num_workers" : 2,

    # ── model ────────────────────────────────
    "feat_dim"    : 256,
    "dropout"     : 0.4,

    # ── training ─────────────────────────────
    "batch_size"     : 16,      # larger batch = more stable gradients
    "weight_decay"   : 1e-4,

    # stage 1 — heads only (backbone frozen)
    # Higher LR is safe because backbone is frozen
    "epochs_stage1"  : 20,
    "lr_stage1"      : 1e-3,    # 3x higher — heads learn faster

    # stage 2 — full fine-tune
    "epochs_stage2"  : 35,
    "lr_stage2"      : 3e-4,
    "lr_backbone"    : 1e-5,    # 30x lower than heads for pretrained layers
}

if __name__ == "__main__":
    train(CONFIG)
