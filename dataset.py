import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ──────────────────────────────────────────────
#  Augmentation pipelines
# ──────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 112):
    """
    Returns torchvision transforms for each split.
    - train : heavy augmentation
    - val/test : only resize + normalize
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class WLASLBodyPartDataset(Dataset):
    """
    Loads multi-stream crops (face, left_hand, right_hand) for WLASL.

    Expected directory layout
    ─────────────────────────
    crops/
      {split}/
        frames/
          {class_name}/
            {video_id}/
              face/          ← frame_0000.jpg, frame_0001.jpg, …
              left_hand/
              right_hand/

    Args
    ────
    root       : path to `crops/` directory
    split      : "train" | "val" | "test"
    num_frames : number of frames to uniformly sample per video
    img_size   : spatial size fed to the backbone
    """

    PARTS = ["face", "left_hand", "right_hand"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
        img_size: int = 112,
    ):
        self.root       = Path(root)
        self.split      = split
        self.num_frames = num_frames
        self.transform  = get_transforms(split, img_size)

        # ── collect (video_dir, label) pairs ──────────────────────────
        frames_dir = self.root / split / "frames"
        if not frames_dir.exists():
            raise FileNotFoundError(f"Directory not found: {frames_dir}")

        class_names = sorted(
            d.name for d in frames_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes  = len(class_names)

        self.samples: list[tuple[Path, int]] = []
        for cls in class_names:
            cls_dir = frames_dir / cls
            for vid_dir in sorted(cls_dir.iterdir()):
                if vid_dir.is_dir():
                    # only keep videos that have ALL three body parts
                    if all((vid_dir / p).is_dir() for p in self.PARTS):
                        self.samples.append((vid_dir, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in {frames_dir}. "
                "Make sure each video folder has face/, left_hand/, right_hand/ sub-folders."
            )

        print(f"[{split}] {len(self.samples)} videos | "
              f"{self.num_classes} classes | "
              f"{self.num_frames} frames per video")

    # ── helpers ───────────────────────────────────────────────────────

    def _sample_frame_indices(self, total: int) -> list[int]:
        """Uniformly sample `num_frames` indices from [0, total)."""
        if total <= self.num_frames:
            # repeat last frame if clip is too short
            indices = list(range(total))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            return indices
        step = total / self.num_frames
        return [int(i * step) for i in range(self.num_frames)]

    def _load_part_frames(self, vid_dir: Path, part: str) -> torch.Tensor:
        """
        Load `num_frames` images from `vid_dir/part/` and stack them.

        Returns
        ───────
        Tensor of shape (T, C, H, W)
        """
        part_dir = vid_dir / part
        frame_files = sorted(part_dir.glob("*.jpg")) + sorted(part_dir.glob("*.png"))

        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in {part_dir}")

        indices = self._sample_frame_indices(len(frame_files))

        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx]).convert("RGB")
            # NOTE: for train split, apply the *same* random state per
            # body-part so spatial flips are consistent across streams
            frames.append(self.transform(img))

        return torch.stack(frames)   # (T, C, H, W)

    # ── Dataset interface ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        vid_dir, label = self.samples[idx]

        # ── fix random seed per sample so augmentations are consistent
        #    across all three body-part streams ─────────────────────────
        seed = random.randint(0, 2**31)

        part_tensors = {}
        for part in self.PARTS:
            random.seed(seed)
            torch.manual_seed(seed)
            part_tensors[part] = self._load_part_frames(vid_dir, part)

        # shapes: each (T, C, H, W)
        return (
            part_tensors["face"],        # (T, C, H, W)
            part_tensors["left_hand"],   # (T, C, H, W)
            part_tensors["right_hand"],  # (T, C, H, W)
            torch.tensor(label, dtype=torch.long),
        )
