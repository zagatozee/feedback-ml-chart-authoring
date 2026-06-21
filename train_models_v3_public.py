"""
feedBack ML Chart Authoring Models - Training Script v3

Dual-encoder architecture:
  - Main encoder:       D_MODEL=256, N_LAYERS=4  (note sequence)
  - Transition encoder: D_MODEL=128, N_LAYERS=2  (technique transition features)
  - Combined head classifies over concatenated pooled representations

Four models trained sequentially:
  1. Anchor placement    — all types, 100 epochs
  2. Lead fingering      — Lead only, 100 epochs
  3. Rhythm fingering    — Rhythm only, 100 epochs
  4. Bass string         — Bass only, 30 epochs

Input:
  training_data_v3_anchor.jsonl   (anchor-zone records, all arrangement types)
  training_data_v3_lead.jsonl     (lead guitar records)
  training_data_v3_rhythm.jsonl   (rhythm guitar records)
  training_data_v3_bass.jsonl     (bass records)
  
  Generate training data using extract_training_data_v3.py against your
  own licensed chart library.

Output:
  models/anchor_model_v3.pt
  models/fingering_lead_v3.pt
  models/fingering_rhythm_v3.pt
  models/fingering_bass_v3.pt
  models/training_v3_meta.json

Run: python train_models_v3.py
"""

import json
import os
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = r"./"
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DATA = {
    "anchor": os.path.join(BASE_DIR, "training_data_v3_anchor.jsonl"),
    "lead":   os.path.join(BASE_DIR, "training_data_v3_lead.jsonl"),
    "rhythm": os.path.join(BASE_DIR, "training_data_v3_rhythm.jsonl"),
    "bass":   os.path.join(BASE_DIR, "training_data_v3_bass.jsonl"),
}

# ─── Hyperparameters ──────────────────────────────────────────────────────────
MAX_NOTES        = 32
NOTE_FEAT_DIM    = 20
TRANS_FEAT_DIM   = 32    # transition feature vector size
CONTEXT_DIM      = 42    # same as v2 (10 base + 2*2*8 neighbour)

# Main encoder (note sequence)
D_MODEL_MAIN     = 256
N_HEADS_MAIN     = 8
N_LAYERS_MAIN    = 4
D_FF_MAIN        = 512

# Transition encoder (lighter)
D_MODEL_TRANS    = 128
N_HEADS_TRANS    = 4
N_LAYERS_TRANS   = 2
D_FF_TRANS       = 256

DROPOUT          = 0.15

NUM_FRET_CLASSES    = 23
FINGER_CLASSES      = 6
FINGER_OFFSET       = 1
BASS_STRING_CLASSES = 4

BATCH_SIZE       = 128    # smaller batch due to larger model + dual encoder
EPOCHS_MAIN      = 100
EPOCHS_BASS      = 30
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 5
VAL_SPLIT        = 0.1
EARLY_STOP       = 15
SEED             = 42

TECHNIQUE_LIST = [
    "palmMute", "hammerOn", "pullOff", "hopo", "bend",
    "vibrato", "tremolo", "harmonicPinch", "harmonic",
    "slideUnpitchTo", "slideTo", "linkNext",
]

# ─── Note Encoding (same as v2) ───────────────────────────────────────────────

def encode_note(note: dict) -> list:
    fret    = note.get("fret", 0)
    string  = note.get("string", 0)
    sustain = min(note.get("sustain", 0), 4.0)
    feats = [
        fret / 22.0,
        string / 5.0,
        sustain / 4.0,
        1.0 if fret == 0 else 0.0,
        1.0 if fret <= 4 else 0.0,
        1.0 if 5 <= fret <= 9 else 0.0,
        1.0 if 10 <= fret <= 14 else 0.0,
        1.0 if fret >= 15 else 0.0,
    ]
    for tech in TECHNIQUE_LIST:
        feats.append(1.0 if note.get(tech, 0) else 0.0)
    assert len(feats) == NOTE_FEAT_DIM
    return feats


def encode_transition(tr: dict) -> list:
    """
    Encode a transition record into a fixed-length feature vector.
    32 features covering slides, HO/PO, HOFN, taps, and general transitions.
    """
    def f(key, default=0.0, scale=1.0):
        v = tr.get(key, default)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        return float(v) * scale

    feats = [
        # General transition (8 features)
        f("prev_note_fret", -1) / 22.0,
        f("prev_note_string", -1) / 5.0,
        min(f("prev_note_time_delta", 0), 2.0) / 2.0,
        f("next_note_fret", -1) / 22.0,
        f("next_note_string", -1) / 5.0,
        min(f("fret_jump_from_prev"), 22.0) / 22.0,
        f("is_position_shift"),
        f("is_string_cross"),

        # Anchor change (4 features)
        f("anchor_changes_here"),
        f("anchor_fret_before", -1) / 22.0,
        f("anchor_fret_after", -1) / 22.0,
        min(f("anchor_shift_distance"), 22.0) / 22.0,

        # Slide (6 features)
        f("slide_start_fret", -1) / 22.0,
        f("slide_end_fret", -1) / 22.0,
        max(-1.0, min(1.0, f("slide_fret_delta") / 11.0)),
        f("slide_is_upward"),
        f("slide_crosses_anchor"),
        f("slide_anchor_change_follows"),

        # HO/PO (6 features)
        max(-1.0, min(1.0, f("hopo_fret_delta") / 4.0)),
        f("hopo_is_ascending"),
        f("hopo_direction_consistent"),
        min(f("hopo_stretch"), 8.0) / 8.0,
        f("hopo_within_anchor"),
        f("hopo_crosses_anchor"),

        # HOFN (4 features)
        f("is_hofn"),
        f("anchor_established_before_hofn"),
        min(f("notes_in_anchor_before_hofn"), 20.0) / 20.0,
        f("hofn_finger_required", -1) / 4.0,

        # Tap (4 features)
        f("tap_fret", -1) / 22.0,
        min(f("tap_distance_from_frethand"), 12.0) / 12.0,
        f("anchor_excludes_tap_fret"),
        f("anchor_covers_frethand_only"),
    ]

    assert len(feats) == TRANS_FEAT_DIM, f"Expected {TRANS_FEAT_DIM}, got {len(feats)}"
    return feats


def encode_zone(record: dict):
    notes = list(record.get("notes", []))
    for chord in record.get("chords", []):
        for cn in chord.get("chord_notes", []):
            notes.append(cn)
    notes = sorted(notes, key=lambda n: n.get("time", 0))[:MAX_NOTES]
    n_real = len(notes)

    matrix = np.zeros((MAX_NOTES, NOTE_FEAT_DIM), dtype=np.float32)
    for i, note in enumerate(notes):
        matrix[i] = encode_note(note)

    pad_mask = np.ones(MAX_NOTES, dtype=bool)
    pad_mask[:n_real] = False
    if n_real == 0:
        pad_mask[0] = False

    return matrix, pad_mask


def encode_transition_matrix(record: dict):
    """Encode transition records into (MAX_NOTES, TRANS_FEAT_DIM) matrix."""
    transitions = record.get("transitions", [])
    transitions = transitions[:MAX_NOTES]
    n_real = len(transitions)

    matrix = np.zeros((MAX_NOTES, TRANS_FEAT_DIM), dtype=np.float32)
    for i, tr in enumerate(transitions):
        matrix[i] = encode_transition(tr)

    # Pad mask mirrors note pad mask
    pad_mask = np.ones(MAX_NOTES, dtype=bool)
    pad_mask[:n_real] = False
    if n_real == 0:
        pad_mask[0] = False

    return matrix, pad_mask


def encode_context(record: dict) -> list:
    """Same as v2 context encoding."""
    stats  = record.get("stats", {})
    anchor = record.get("anchor", {})
    arr    = record.get("arrangement", "Lead")
    fret_min = stats.get("fret_min", 0)
    fret_max = stats.get("fret_max", 0)

    NEIGHBOUR_DIM = 8
    N_NEIGHBOURS  = 2

    base = [
        min(stats.get("note_count", 0), 64) / 64.0,
        min(stats.get("chord_count", 0), 32) / 32.0,
        fret_min / 22.0,
        fret_max / 22.0,
        (fret_max - fret_min) / 22.0,
        anchor.get("fret", 0) / 22.0,
        min(record.get("avg_bpm", 120) or 120, 300) / 300.0,
        min(anchor.get("duration", 1.0), 10.0) / 10.0,
        1.0 if arr == "Lead"   else 0.0,
        1.0 if arr == "Bass"   else 0.0,
    ]

    def encode_neighbour(zone):
        if not zone:
            return [0.0] * NEIGHBOUR_DIM
        s = zone.get("summary", {})
        af = zone.get("anchor_fret", 0)
        aw = zone.get("anchor_width", 4)
        fm = s.get("fret_min", 0)
        fx = s.get("fret_max", 0)
        return [
            af / 22.0,
            aw / 8.0,
            min(s.get("note_count", 0), 64) / 64.0,
            min(s.get("chord_count", 0), 32) / 32.0,
            fm / 22.0,
            fx / 22.0,
            (fx - fm) / 22.0,
            1.0 if s.get("techniques") else 0.0,
        ]

    prev_zones = record.get("prev_zones", [])
    next_zones = record.get("next_zones", [])
    for j in range(N_NEIGHBOURS):
        base.extend(encode_neighbour(prev_zones[-(j+1)] if j < len(prev_zones) else {}))
    for j in range(N_NEIGHBOURS):
        base.extend(encode_neighbour(next_zones[j] if j < len(next_zones) else {}))

    return base

ACTUAL_CONTEXT_DIM = len(encode_context({}))

# ─── Label Encoding ───────────────────────────────────────────────────────────

def encode_guitar_labels(record: dict):
    templates = record.get("chord_templates", {})
    if not templates:
        return None
    primary = None
    best_count = -1
    for tid, tmpl in templates.items():
        rc = tmpl.get("ref_count", 1)
        if tmpl.get("is_primary", False) or rc > best_count:
            primary = tmpl
            best_count = rc
    if primary is None:
        primary = next(iter(templates.values()))
    fingers = primary.get("fingers", [-1] * 6)
    return np.array([
        max(0, min(FINGER_CLASSES - 1, f + FINGER_OFFSET))
        for f in fingers
    ], dtype=np.int64)


def encode_bass_labels(record: dict) -> np.ndarray:
    notes  = sorted(record.get("notes", []), key=lambda n: n.get("time", 0))[:MAX_NOTES]
    labels = np.full(MAX_NOTES, -1, dtype=np.int64)
    for i, note in enumerate(notes):
        s = note.get("string", 0)
        labels[i] = max(0, min(BASS_STRING_CLASSES - 1, s))
    return labels

# ─── Datasets ─────────────────────────────────────────────────────────────────

class AnchorDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        notes, pad_mask   = encode_zone(rec)
        trans, trans_mask = encode_transition_matrix(rec)
        context = encode_context(rec)
        label   = max(0, min(rec["anchor"]["fret"], NUM_FRET_CLASSES - 1))
        return {
            "notes":      torch.tensor(notes,      dtype=torch.float32),
            "pad_mask":   torch.tensor(pad_mask,   dtype=torch.bool),
            "trans":      torch.tensor(trans,      dtype=torch.float32),
            "trans_mask": torch.tensor(trans_mask, dtype=torch.bool),
            "context":    torch.tensor(context,    dtype=torch.float32),
            "label":      torch.tensor(label,      dtype=torch.long),
        }


class GuitarDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        notes, pad_mask   = encode_zone(rec)
        trans, trans_mask = encode_transition_matrix(rec)
        context = encode_context(rec)
        labels  = encode_guitar_labels(rec)
        return {
            "notes":      torch.tensor(notes,      dtype=torch.float32),
            "pad_mask":   torch.tensor(pad_mask,   dtype=torch.bool),
            "trans":      torch.tensor(trans,      dtype=torch.float32),
            "trans_mask": torch.tensor(trans_mask, dtype=torch.bool),
            "context":    torch.tensor(context,    dtype=torch.float32),
            "labels":     torch.tensor(labels,     dtype=torch.long),
        }


class BassDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        notes, pad_mask   = encode_zone(rec)
        trans, trans_mask = encode_transition_matrix(rec)
        context = encode_context(rec)
        labels  = encode_bass_labels(rec)
        return {
            "notes":      torch.tensor(notes,      dtype=torch.float32),
            "pad_mask":   torch.tensor(pad_mask,   dtype=torch.bool),
            "trans":      torch.tensor(trans,      dtype=torch.float32),
            "trans_mask": torch.tensor(trans_mask, dtype=torch.bool),
            "context":    torch.tensor(context,    dtype=torch.float32),
            "labels":     torch.tensor(labels,     dtype=torch.long),
        }

# ─── Model ────────────────────────────────────────────────────────────────────

class DualEncoderModel(nn.Module):
    """
    Dual-encoder chart authoring model.

    Main encoder (D_MODEL=256, N_LAYERS=4) processes note sequence.
    Transition encoder (D_MODEL=128, N_LAYERS=2) processes technique transitions.
    Both are pooled and concatenated before the task-specific head.

    mode: 'anchor' | 'guitar' | 'bass'
    """
    def __init__(self, mode="anchor", ctx_dim=None):
        super().__init__()
        self.mode   = mode
        ctx_dim     = ctx_dim or ACTUAL_CONTEXT_DIM

        # ── Main note encoder ────────────────────────────────────────────────
        self.note_proj = nn.Linear(NOTE_FEAT_DIM, D_MODEL_MAIN)
        self.note_pos  = nn.Embedding(MAX_NOTES, D_MODEL_MAIN)
        main_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL_MAIN, nhead=N_HEADS_MAIN,
            dim_feedforward=D_FF_MAIN, dropout=DROPOUT,
            batch_first=True, norm_first=True,
        )
        self.main_encoder = nn.TransformerEncoder(main_layer, num_layers=N_LAYERS_MAIN)

        # ── Transition encoder ───────────────────────────────────────────────
        self.trans_proj = nn.Linear(TRANS_FEAT_DIM, D_MODEL_TRANS)
        self.trans_pos  = nn.Embedding(MAX_NOTES, D_MODEL_TRANS)
        trans_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL_TRANS, nhead=N_HEADS_TRANS,
            dim_feedforward=D_FF_TRANS, dropout=DROPOUT,
            batch_first=True, norm_first=True,
        )
        self.trans_encoder = nn.TransformerEncoder(trans_layer, num_layers=N_LAYERS_TRANS)

        # ── Context projection ───────────────────────────────────────────────
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Combined dimension: main pool + trans pool + ctx
        combined_dim = D_MODEL_MAIN + D_MODEL_TRANS + 64

        # ── Task heads ───────────────────────────────────────────────────────
        if mode == "anchor":
            self.head = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(256, NUM_FRET_CLASSES),
            )
        elif mode == "guitar":
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(combined_dim, 128),
                    nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(128, FINGER_CLASSES),
                )
                for _ in range(6)
            ])
        elif mode == "bass":
            self.head = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(128, BASS_STRING_CLASSES),
            )

    def pool(self, x, pad_mask):
        """Mean pool over non-padded positions."""
        real = (~pad_mask).float().unsqueeze(-1)
        return (x * real).sum(1) / real.sum(1).clamp(min=1)

    def forward(self, notes, pad_mask, trans, trans_mask, context):
        B, S, _ = notes.shape

        # Main encoder
        x = self.note_proj(notes)
        pos = torch.arange(S, device=notes.device).unsqueeze(0).expand(B, -1)
        x = x + self.note_pos(pos)
        x = self.main_encoder(x, src_key_padding_mask=pad_mask)
        main_pooled = self.pool(x, pad_mask)

        # Transition encoder
        t = self.trans_proj(trans)
        t = t + self.trans_pos(pos)
        t = self.trans_encoder(t, src_key_padding_mask=trans_mask)
        trans_pooled = self.pool(t, trans_mask)

        # Context
        ctx = self.ctx_proj(context)

        # Combine
        combined = torch.cat([main_pooled, trans_pooled, ctx], dim=-1)

        if self.mode == "anchor":
            return self.head(combined)
        elif self.mode == "guitar":
            return torch.stack([h(combined) for h in self.head], dim=1)
        elif self.mode == "bass":
            return self.head(combined)

# ─── LR Schedule ──────────────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_jsonl(path, filter_fn=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(path)}"):
            try:
                rec = json.loads(line)
                if filter_fn is None or filter_fn(rec):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def split_by_song(records, val_split=0.1, seed=42):
    random.seed(seed)
    by_song = defaultdict(list)
    for rec in records:
        key = f"{rec['artist']}||{rec['song']}"
        by_song[key].append(rec)
    songs = list(by_song.keys())
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * val_split))
    val_songs   = set(songs[:n_val])
    train_songs = set(songs[n_val:])
    train = [r for k in train_songs for r in by_song[k]]
    val   = [r for k in val_songs   for r in by_song[k]]
    print(f"  Train: {len(train):,} from {len(train_songs):,} songs")
    print(f"  Val:   {len(val):,} from {len(val_songs):,} songs")
    return train, val

# ─── Training Functions ───────────────────────────────────────────────────────

def run_anchor_epoch(model, loader, optimizer, criterion, device, scaler, train=True):
    model.train() if train else model.eval()
    total_loss = correct = within1 = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if train else "Val  ", leave=False):
            notes      = batch["notes"].to(device)
            pad_mask   = batch["pad_mask"].to(device)
            trans      = batch["trans"].to(device)
            trans_mask = batch["trans_mask"].to(device)
            context    = batch["context"].to(device)
            labels     = batch["label"].to(device)
            if train:
                optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(notes, pad_mask, trans, trans_mask, context)
                loss   = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            preds      = logits.argmax(-1)
            correct    += (preds == labels).sum().item()
            within1    += (torch.abs(preds - labels) <= 1).sum().item()
            total      += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    return total_loss / total, correct / total, within1 / total


def run_guitar_epoch(model, loader, optimizer, criterion, device, scaler, train=True):
    model.train() if train else model.eval()
    total_loss = exact = total = 0
    per_string = torch.zeros(6)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if train else "Val  ", leave=False):
            notes      = batch["notes"].to(device)
            pad_mask   = batch["pad_mask"].to(device)
            trans      = batch["trans"].to(device)
            trans_mask = batch["trans_mask"].to(device)
            context    = batch["context"].to(device)
            labels     = batch["labels"].to(device)
            if train:
                optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(notes, pad_mask, trans, trans_mask, context)
                loss   = sum(criterion(logits[:, s, :], labels[:, s]) for s in range(6)) / 6.0
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            preds = logits.argmax(-1)
            exact += (preds == labels).all(-1).sum().item()
            for s in range(6):
                per_string[s] += (preds[:, s] == labels[:, s]).sum().item()
            total      += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    return total_loss / total, exact / total, (per_string / total).tolist()


def run_bass_epoch(model, loader, optimizer, criterion, device, scaler, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if train else "Val  ", leave=False):
            notes      = batch["notes"].to(device)
            pad_mask   = batch["pad_mask"].to(device)
            trans      = batch["trans"].to(device)
            trans_mask = batch["trans_mask"].to(device)
            context    = batch["context"].to(device)
            labels     = batch["labels"].to(device)
            if train:
                optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(notes, pad_mask, trans, trans_mask, context)
                valid  = labels >= 0
                loss   = criterion(logits[valid], labels[valid])
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            preds      = logits.argmax(-1)
            correct    += (preds[valid] == labels[valid]).sum().item()
            total      += valid.sum().item()
            total_loss += loss.item() * valid.sum().item()
    return total_loss / max(total, 1), correct / max(total, 1)

# ─── Generic Training Loop ────────────────────────────────────────────────────

def train_model(name, model, train_loader, val_loader, mode, device,
                epochs, weights=None, out_path=None):
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, epochs)
    scaler    = torch.amp.GradScaler("cuda")

    best_val  = 0.0
    patience  = 0
    history   = []
    out_path  = out_path or os.path.join(MODELS_DIR, f"{name}.pt")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*65}")
    print(f"TRAINING: {name}  ({n_params:,} parameters)")
    print(f"{'='*65}")

    if mode == "anchor":
        print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'ValLoss':>8} {'ValAcc':>7} {'Val±1':>7} {'LR':>9}")
        print("-" * 62)
    else:
        print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'ValLoss':>8} {'ValExact':>9} {'LR':>9}")
        print("-" * 60)

    for epoch in range(1, epochs + 1):
        if mode == "anchor":
            t_loss, t_acc, _    = run_anchor_epoch(model, train_loader, optimizer, criterion, device, scaler, True)
            v_loss, v_acc, v_w1 = run_anchor_epoch(model, val_loader,   optimizer, criterion, device, scaler, False)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>6} {t_loss:>8.4f} {t_acc:>6.1%} {v_loss:>8.4f} {v_acc:>6.1%} {v_w1:>6.1%} {lr:>9.2e}")
            metric = v_acc
            history.append({"epoch": epoch, "train_loss": t_loss, "train_acc": t_acc,
                             "val_loss": v_loss, "val_acc": v_acc, "val_within1": v_w1})

        elif mode == "guitar":
            t_loss, t_acc, _          = run_guitar_epoch(model, train_loader, optimizer, criterion, device, scaler, True)
            v_loss, v_exact, v_per_s  = run_guitar_epoch(model, val_loader,   optimizer, criterion, device, scaler, False)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>6} {t_loss:>8.4f} {t_acc:>6.1%} {v_loss:>8.4f} {v_exact:>8.1%} {lr:>9.2e}")
            metric = v_exact
            history.append({"epoch": epoch, "train_loss": t_loss, "train_acc": t_acc,
                             "val_loss": v_loss, "val_exact": v_exact, "val_per_string": v_per_s})

        elif mode == "bass":
            t_loss, t_acc = run_bass_epoch(model, train_loader, optimizer, criterion, device, scaler, True)
            v_loss, v_acc = run_bass_epoch(model, val_loader,   optimizer, criterion, device, scaler, False)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>6} {t_loss:>8.4f} {t_acc:>6.1%} {v_loss:>8.4f} {v_acc:>8.1%} {lr:>9.2e}")
            metric = v_acc
            history.append({"epoch": epoch, "train_loss": t_loss, "train_acc": t_acc,
                             "val_loss": v_loss, "val_acc": v_acc})

        if metric > best_val:
            best_val = metric
            patience = 0
            torch.save(model.state_dict(), out_path)
            print(f"         ↑ Saved (best={metric:.1%})")
            if mode == "guitar":
                print(f"           Per-string: {[f'{a:.1%}' for a in v_per_s]}")
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\n  Best: {best_val:.1%}  →  {out_path}")
    return best_val, history

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Context dim: {ACTUAL_CONTEXT_DIM}")

    all_results = {}
    all_history = {}

    # ── 1. Anchor ─────────────────────────────────────────────────────────────
    print("\n>>> Loading anchor data...")
    anchor_recs    = load_jsonl(DATA["anchor"])
    train_a, val_a = split_by_song(anchor_recs, VAL_SPLIT, SEED)
    del anchor_recs

    fret_counts = defaultdict(int)
    for rec in train_a:
        fret_counts[rec["anchor"]["fret"]] += 1
    total_a = sum(fret_counts.values())
    weights_a = torch.zeros(NUM_FRET_CLASSES)
    for fret, count in fret_counts.items():
        if 0 <= fret < NUM_FRET_CLASSES:
            weights_a[fret] = total_a / (NUM_FRET_CLASSES * count)

    dl_a_train = DataLoader(AnchorDataset(train_a), BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    dl_a_val   = DataLoader(AnchorDataset(val_a),   BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    m = DualEncoderModel(mode="anchor", ctx_dim=ACTUAL_CONTEXT_DIM).to(device)
    best, hist = train_model("anchor_model_v3", m, dl_a_train, dl_a_val,
                             "anchor", device, EPOCHS_MAIN, weights_a,
                             os.path.join(MODELS_DIR, "anchor_model_v3.pt"))
    all_results["anchor"] = best
    all_history["anchor"] = hist
    del train_a, val_a, dl_a_train, dl_a_val, m

    # ── 2. Lead ───────────────────────────────────────────────────────────────
    print("\n>>> Loading Lead data...")
    lead_recs      = load_jsonl(DATA["lead"], filter_fn=lambda r: bool(r.get("chord_templates")))
    train_l, val_l = split_by_song(lead_recs, VAL_SPLIT, SEED)
    del lead_recs

    dl_l_train = DataLoader(GuitarDataset(train_l), BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    dl_l_val   = DataLoader(GuitarDataset(val_l),   BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    m = DualEncoderModel(mode="guitar", ctx_dim=ACTUAL_CONTEXT_DIM).to(device)
    best, hist = train_model("fingering_lead_v3", m, dl_l_train, dl_l_val,
                             "guitar", device, EPOCHS_MAIN, None,
                             os.path.join(MODELS_DIR, "fingering_lead_v3.pt"))
    all_results["lead"] = best
    all_history["lead"] = hist
    del train_l, val_l, dl_l_train, dl_l_val, m

    # ── 3. Rhythm ─────────────────────────────────────────────────────────────
    print("\n>>> Loading Rhythm data...")
    rhythm_recs    = load_jsonl(DATA["rhythm"], filter_fn=lambda r: bool(r.get("chord_templates")))
    train_r, val_r = split_by_song(rhythm_recs, VAL_SPLIT, SEED)
    del rhythm_recs

    dl_r_train = DataLoader(GuitarDataset(train_r), BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    dl_r_val   = DataLoader(GuitarDataset(val_r),   BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    m = DualEncoderModel(mode="guitar", ctx_dim=ACTUAL_CONTEXT_DIM).to(device)
    best, hist = train_model("fingering_rhythm_v3", m, dl_r_train, dl_r_val,
                             "guitar", device, EPOCHS_MAIN, None,
                             os.path.join(MODELS_DIR, "fingering_rhythm_v3.pt"))
    all_results["rhythm"] = best
    all_history["rhythm"] = hist
    del train_r, val_r, dl_r_train, dl_r_val, m

    # ── 4. Bass ───────────────────────────────────────────────────────────────
    print("\n>>> Loading Bass data...")
    bass_recs      = load_jsonl(DATA["bass"], filter_fn=lambda r: bool(r.get("notes")))
    train_b, val_b = split_by_song(bass_recs, VAL_SPLIT, SEED)
    del bass_recs

    dl_b_train = DataLoader(BassDataset(train_b), BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    dl_b_val   = DataLoader(BassDataset(val_b),   BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    m = DualEncoderModel(mode="bass", ctx_dim=ACTUAL_CONTEXT_DIM).to(device)
    best, hist = train_model("fingering_bass_v3", m, dl_b_train, dl_b_val,
                             "bass", device, EPOCHS_BASS, None,
                             os.path.join(MODELS_DIR, "fingering_bass_v3.pt"))
    all_results["bass"] = best
    all_history["bass"] = hist

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "version":              "v3",
        "architecture":         "DualEncoder",
        "d_model_main":         D_MODEL_MAIN,
        "n_layers_main":        N_LAYERS_MAIN,
        "d_model_trans":        D_MODEL_TRANS,
        "n_layers_trans":       N_LAYERS_TRANS,
        "note_feat_dim":        NOTE_FEAT_DIM,
        "trans_feat_dim":       TRANS_FEAT_DIM,
        "context_dim":          ACTUAL_CONTEXT_DIM,
        "max_notes":            MAX_NOTES,
        "num_fret_classes":     NUM_FRET_CLASSES,
        "finger_classes":       FINGER_CLASSES,
        "finger_offset":        FINGER_OFFSET,
        "bass_string_classes":  BASS_STRING_CLASSES,
        "technique_list":       TECHNIQUE_LIST,
        "best_results":         {k: float(v) for k, v in all_results.items()},
        "history":              all_history,
    }
    meta_path = os.path.join(MODELS_DIR, "training_v3_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*65)
    print("ALL TRAINING COMPLETE")
    print("="*65)
    for name, acc in all_results.items():
        print(f"  {name:>10}: {acc:.1%}")
    print(f"\n  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
