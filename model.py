import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ──────────────────────────────────────────────
#  Stream encoder (shared for left/right hand)
# ──────────────────────────────────────────────

class HandEncoder(nn.Module):
    """
    EfficientNet-B2 backbone that encodes a single frame crop.
    The classifier head is replaced by a projection layer.

    Output: (B, feat_dim)
    """

    def __init__(self, feat_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b2(weights=weights)

        # remove the original classifier
        in_features = backbone.classifier[1].in_features   # 1408 for B2
        backbone.classifier = nn.Identity()

        self.backbone  = backbone
        self.proj      = nn.Sequential(
            nn.Linear(in_features, feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, feat_dim)"""
        feat = self.backbone(x)          # (B, 1408)
        return self.proj(feat)           # (B, feat_dim)


class FaceEncoder(nn.Module):
    """
    Lightweight MobileNetV3-Small backbone for face crops.
    Separate weights because face features differ from hand features.

    MobileNetV3-Small architecture:
      features → avgpool → flatten → (576,) → classifier[0] Linear(576→1024) → ...
    We extract the 576-dim vector BEFORE the classifier by using
    features + avgpool directly, so in_features = 576 always.

    Output: (B, feat_dim)
    """

    IN_FEATURES = 576   # hardcoded — MobileNetV3-Small is fixed

    def __init__(self, feat_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = mobilenet_v3_small(weights=weights)

        # Keep only the feature extractor + pooling; discard classifier entirely
        self.features = net.features
        self.avgpool  = net.avgpool   # AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(self.IN_FEATURES, feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, feat_dim)"""
        x = self.features(x)        # (B, 96, H', W')
        x = self.avgpool(x)         # (B, 96, 1, 1)  — wait, actual channels=576 after features
        x = x.flatten(1)            # (B, 576)
        return self.proj(x)         # (B, feat_dim)


# ──────────────────────────────────────────────
#  Temporal Transformer
# ──────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Small Transformer encoder that aggregates T frame features
    into a single video-level representation via mean pooling.

    Input : (B, T, d_model)
    Output: (B, d_model)
    """

    def __init__(
        self,
        d_model: int    = 512,
        nhead: int      = 4,
        num_layers: int = 2,
        dropout: float  = 0.1,
        max_len: int    = 32,
    ):
        super().__init__()

        # learnable positional embeddings
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= d_model * 4,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,    # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, d_model)"""
        B, T, _ = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        x    = x + self.pos_emb(pos)
        x    = self.encoder(x)          # (B, T, d_model)
        x    = self.norm(x)
        return x.mean(dim=1)            # mean pool over T → (B, d_model)


# ──────────────────────────────────────────────
#  Full multi-stream model
# ──────────────────────────────────────────────

class MultiStreamSLR(nn.Module):
    """
    Multi-stream Sign Language Recognition model.

    Streams
    ───────
    • left_hand  )
    • right_hand ) → shared HandEncoder (EfficientNet-B2)
    • face         → separate FaceEncoder (MobileNetV3-Small)

    Pipeline per stream
    ───────────────────
    (B, T, C, H, W) → flatten T into batch → encode frame →
    reshape → concat → TemporalTransformer → FC classifier

    Args
    ────
    num_classes : number of sign classes (100 for WLASL-100)
    feat_dim    : per-stream feature dimension (default 256)
    num_frames  : T, used only for pos_emb max_len
    dropout     : classifier dropout probability
    pretrained  : whether to use ImageNet pretrained backbones
    freeze_backbone : freeze CNN weights during stage-1 training
    """

    def __init__(
        self,
        num_classes    : int   = 100,
        feat_dim       : int   = 256,
        num_frames     : int   = 16,
        dropout        : float = 0.5,
        pretrained     : bool  = True,
        freeze_backbone: bool  = True,
    ):
        super().__init__()

        self.num_frames = num_frames
        fused_dim = feat_dim * 3         # concat of 3 streams

        # ── encoders ──────────────────────────────────────────────────
        self.hand_encoder = HandEncoder(feat_dim, pretrained)
        self.face_encoder = FaceEncoder(feat_dim, pretrained)

        # ── temporal aggregation ──────────────────────────────────────
        self.temporal = TemporalTransformer(
            d_model    = fused_dim,
            nhead      = 4,
            num_layers = 2,
            dropout    = 0.1,
            max_len    = num_frames + 4,
        )

        # ── classifier head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbones()

    # ── public API ────────────────────────────────────────────────────

    def freeze_backbones(self):
        """Freeze all CNN backbone parameters (stage-1 training)."""
        for param in self.hand_encoder.backbone.parameters():
            param.requires_grad = False
        for param in self.face_encoder.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbones FROZEN — training heads only.")

    def unfreeze_backbones(self):
        """Unfreeze backbones for end-to-end fine-tuning (stage-2)."""
        for param in self.hand_encoder.backbone.parameters():
            param.requires_grad = True
        for param in self.face_encoder.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbones UNFROZEN — full fine-tuning.")

    # ── forward ───────────────────────────────────────────────────────

    def _encode_stream(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        """
        Encode one body-part stream.

        x : (B, T, C, H, W)
        returns: (B, T, feat_dim)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)           # (B*T, C, H, W)
        feat   = encoder(x_flat)                   # (B*T, feat_dim)
        return feat.view(B, T, -1)                 # (B, T, feat_dim)

    def forward(
        self,
        face      : torch.Tensor,    # (B, T, C, H, W)
        left_hand : torch.Tensor,    # (B, T, C, H, W)
        right_hand: torch.Tensor,    # (B, T, C, H, W)
    ) -> torch.Tensor:
        """Returns logits of shape (B, num_classes)."""

        # per-stream spatial encoding
        f_face  = self._encode_stream(face,       self.face_encoder)
        f_left  = self._encode_stream(left_hand,  self.hand_encoder)
        f_right = self._encode_stream(right_hand, self.hand_encoder)

        # fuse by concatenation along feature dim
        fused = torch.cat([f_face, f_left, f_right], dim=-1)  # (B, T, 3*feat_dim)

        # temporal aggregation
        video_feat = self.temporal(fused)          # (B, 3*feat_dim)

        return self.classifier(video_feat)         # (B, num_classes)
