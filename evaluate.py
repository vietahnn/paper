import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import WLASLBodyPartDataset
from model   import MultiStreamSLR


# ──────────────────────────────────────────────
#  Evaluation helpers
# ──────────────────────────────────────────────

def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    _, top_k = logits.topk(k, dim=-1)
    correct  = top_k.eq(labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


@torch.no_grad()
def evaluate_test(
    ckpt_path : str,
    data_root : str,
    num_frames: int = 16,
    img_size  : int = 112,
    batch_size: int = 8,
    num_workers: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset ───────────────────────────────────────────────────────
    # Use val split if test labels are unavailable
    for split in ("test", "val"):
        try:
            ds = WLASLBodyPartDataset(
                root       = data_root,
                split      = split,
                num_frames = num_frames,
                img_size   = img_size,
            )
            print(f"[Evaluate] Using split: {split}")
            break
        except (FileNotFoundError, RuntimeError):
            continue

    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = MultiStreamSLR(
        num_classes     = ds.num_classes,
        num_frames      = num_frames,
        freeze_backbone = False,
        pretrained      = False,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Evaluate] Loaded checkpoint: {ckpt_path}\n")

    # ── inference loop ────────────────────────────────────────────────
    all_logits, all_labels = [], []

    for face, lh, rh, labels in loader:
        face   = face.to(device)
        lh     = lh.to(device)
        rh     = rh.to(device)
        labels = labels.to(device)

        logits = model(face, lh, rh)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # ── metrics ───────────────────────────────────────────────────────
    top1 = top_k_accuracy(all_logits, all_labels, k=1)
    top5 = top_k_accuracy(all_logits, all_labels, k=5)

    print(f"{'─'*40}")
    print(f"  Samples : {len(all_labels)}")
    print(f"  Top-1   : {top1:.4f}  ({top1*100:.2f}%)")
    print(f"  Top-5   : {top5:.4f}  ({top5*100:.2f}%)")
    print(f"{'─'*40}\n")

    # ── per-class accuracy ────────────────────────────────────────────
    preds = all_logits.argmax(dim=-1)
    print("Per-class accuracy (top-10 best / worst):\n")

    per_class = {}
    for c in range(ds.num_classes):
        mask = all_labels == c
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == all_labels[mask]).float().mean().item()
        per_class[ds.idx_to_class[c]] = acc

    sorted_cls = sorted(per_class.items(), key=lambda x: x[1], reverse=True)

    print("  Best 10:")
    for name, acc in sorted_cls[:10]:
        print(f"    {name:<20} {acc:.4f}")

    print("  Worst 10:")
    for name, acc in sorted_cls[-10:]:
        print(f"    {name:<20} {acc:.4f}")

    return {"top1": top1, "top5": top5, "per_class": per_class}


# ──────────────────────────────────────────────
#  Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_test(
        ckpt_path  = "/kaggle/working/checkpoints/best_model.pth",
        data_root  = "/kaggle/input/datasets/nguyenthanhvung/cropdataset/crops",
        num_frames = 16,
        img_size   = 112,
        batch_size = 8,
        num_workers= 2,
    )
