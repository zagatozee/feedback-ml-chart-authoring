"""
feedBack ML Chart Authoring - Inference Engine v1

Takes a Guitar Pro file and produces two arrangement XMLs for A/B comparison:
  {output_dir}/{song}_baseline.xml  — standard conversion output
  {output_dir}/{song}_ml.xml        — ML-enhanced output with intelligent
                                      anchor zones and chord fingerings

The ML engine enhances three aspects of the output:
  1. Anchor placement  — fret-hand position zones minimising position shifts
  2. Chord fingerings  — per-string finger assignments on every chord template
  3. Bass string sel.  — ergonomic string choice per bass note

Usage:
  python inference_engine_v1.py path/to/song.gp5 [output_dir]

  Or import and call convert_gp_file() directly.
"""

import sys
import os
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ─── Configuration ────────────────────────────────────────────────────────────

# Models directory — set via FEEDBACK_MODELS_DIR env var or update this path
MODELS_DIR = Path(os.environ.get("FEEDBACK_MODELS_DIR", 
    str(Path(__file__).parent / "models")))

MODEL_PATHS = {
    "anchor":  MODELS_DIR / "anchor_model_v1.pt",
    "lead":    MODELS_DIR / "fingering_lead_v3.pt",
    "rhythm":  MODELS_DIR / "fingering_rhythm_v3.pt",
    "bass":    MODELS_DIR / "fingering_bass_v3.pt",
}

# Add lib to path for gp2rs
LIB_DIR = Path(os.environ.get("FEEDBACK_LIB_DIR",
    str(Path(__file__).parent / "lib")))
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("rs_inference")

# ─── Model Architecture (must match training scripts) ─────────────────────────

MAX_NOTES        = 32
NOTE_FEAT_DIM    = 20
TRANS_FEAT_DIM   = 32
CONTEXT_DIM      = 42
D_MODEL_MAIN     = 256
N_HEADS_MAIN     = 8
N_LAYERS_MAIN    = 4
D_FF_MAIN        = 512
D_MODEL_TRANS    = 128
N_HEADS_TRANS    = 4
N_LAYERS_TRANS   = 2
D_FF_TRANS       = 256
DROPOUT          = 0.15
NUM_FRET_CLASSES = 23
FINGER_CLASSES   = 6
FINGER_OFFSET    = 1
BASS_STRING_CLASSES = 4

# Anchor model uses v1 architecture (single encoder, smaller capacity)
D_MODEL_ANCHOR   = 128
N_HEADS_ANCHOR   = 4
N_LAYERS_ANCHOR  = 3
D_FF_ANCHOR      = 256
NOTE_FEAT_ANCHOR = 20
CONTEXT_ANCHOR   = 10

TECHNIQUE_LIST = [
    "palmMute", "hammerOn", "pullOff", "hopo", "bend",
    "vibrato", "tremolo", "harmonicPinch", "harmonic",
    "slideUnpitchTo", "slideTo", "linkNext",
]

# ─── Anchor Model (v1 architecture) ───────────────────────────────────────────

class AnchorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.note_proj = nn.Linear(NOTE_FEAT_ANCHOR, D_MODEL_ANCHOR)
        self.pos_embed = nn.Embedding(MAX_NOTES, D_MODEL_ANCHOR)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=D_MODEL_ANCHOR, nhead=N_HEADS_ANCHOR,
            dim_feedforward=D_FF_ANCHOR, dropout=DROPOUT,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS_ANCHOR)
        self.context_proj = nn.Sequential(
            nn.Linear(CONTEXT_ANCHOR, 32), nn.ReLU(), nn.Linear(32, 32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL_ANCHOR + 32, 128),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, NUM_FRET_CLASSES),
        )

    def forward(self, notes, pad_mask, context):
        B, S, _ = notes.shape
        x = self.note_proj(notes)
        pos = torch.arange(S, device=notes.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        real = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * real).sum(1) / real.sum(1).clamp(min=1)
        ctx = self.context_proj(context)
        return self.classifier(torch.cat([pooled, ctx], dim=-1))


# ─── Fingering Model (v3 dual-encoder architecture) ───────────────────────────

class DualEncoderModel(nn.Module):
    def __init__(self, mode="guitar"):
        super().__init__()
        self.mode = mode

        self.note_proj = nn.Linear(NOTE_FEAT_DIM, D_MODEL_MAIN)
        self.note_pos  = nn.Embedding(MAX_NOTES, D_MODEL_MAIN)
        main_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL_MAIN, nhead=N_HEADS_MAIN,
            dim_feedforward=D_FF_MAIN, dropout=DROPOUT,
            batch_first=True, norm_first=True,
        )
        self.main_encoder = nn.TransformerEncoder(main_layer, num_layers=N_LAYERS_MAIN)

        self.trans_proj = nn.Linear(TRANS_FEAT_DIM, D_MODEL_TRANS)
        self.trans_pos  = nn.Embedding(MAX_NOTES, D_MODEL_TRANS)
        trans_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL_TRANS, nhead=N_HEADS_TRANS,
            dim_feedforward=D_FF_TRANS, dropout=DROPOUT,
            batch_first=True, norm_first=True,
        )
        self.trans_encoder = nn.TransformerEncoder(trans_layer, num_layers=N_LAYERS_TRANS)

        self.ctx_proj = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 64), nn.GELU(), nn.Linear(64, 64),
        )

        combined_dim = D_MODEL_MAIN + D_MODEL_TRANS + 64

        if mode == "guitar":
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(combined_dim, 128), nn.GELU(),
                    nn.Dropout(DROPOUT), nn.Linear(128, FINGER_CLASSES),
                )
                for _ in range(6)
            ])
        elif mode == "bass":
            self.bass_head = nn.Sequential(
                nn.Linear(D_MODEL_MAIN + D_MODEL_TRANS, 128),
                nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, BASS_STRING_CLASSES),
            )

    def pool(self, x, pad_mask):
        real = (~pad_mask).float().unsqueeze(-1)
        return (x * real).sum(1) / real.sum(1).clamp(min=1)

    def forward(self, notes, pad_mask, trans, trans_mask, context):
        B, S, _ = notes.shape
        pos = torch.arange(S, device=notes.device).unsqueeze(0).expand(B, -1)

        x = self.note_proj(notes) + self.note_pos(pos)
        x = self.main_encoder(x, src_key_padding_mask=pad_mask)

        t = self.trans_proj(trans) + self.trans_pos(pos)
        t = self.trans_encoder(t, src_key_padding_mask=trans_mask)

        if self.mode == "guitar":
            main_pooled  = self.pool(x, pad_mask)
            trans_pooled = self.pool(t, trans_mask)
            ctx          = self.ctx_proj(context)
            combined     = torch.cat([main_pooled, trans_pooled, ctx], dim=-1)
            return torch.stack([h(combined) for h in self.head], dim=1)  # (B, 6, FINGER_CLASSES)

        elif self.mode == "bass":
            per_pos = torch.cat([x, t], dim=-1)   # (B, S, D_MAIN+D_TRANS)
            return self.bass_head(per_pos)         # (B, S, BASS_STRING_CLASSES)


# ─── Model Loading ────────────────────────────────────────────────────────────

_models = {}

def _load_models(device):
    global _models
    if _models:
        return _models

    log.info("Loading ML models...")

    anchor = AnchorModel()
    anchor.load_state_dict(torch.load(MODEL_PATHS["anchor"], map_location=device, weights_only=True))
    anchor.eval().to(device)
    _models["anchor"] = anchor

    for arr_type in ("lead", "rhythm"):
        m = DualEncoderModel(mode="guitar")
        m.load_state_dict(torch.load(MODEL_PATHS[arr_type], map_location=device, weights_only=True))
        m.eval().to(device)
        _models[arr_type] = m

    bass = DualEncoderModel(mode="bass")
    bass.load_state_dict(torch.load(MODEL_PATHS["bass"], map_location=device, weights_only=True))
    bass.eval().to(device)
    _models["bass"] = bass

    log.info("Models loaded.")
    return _models


# ─── Feature Encoding ─────────────────────────────────────────────────────────

def _encode_rs_note(note) -> list:
    """Encode an RsNote (from gp2rs.py) into a feature vector."""
    fret    = getattr(note, "fret", 0)
    string  = getattr(note, "string", 0)
    sustain = min(getattr(note, "sustain", 0.0), 4.0)

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
    tech_attrs = {
        "palmMute": "palm_mute", "hammerOn": "hammer_on", "pullOff": "pull_off",
        "hopo": None, "bend": "bend", "vibrato": "vibrato", "tremolo": "tremolo",
        "harmonicPinch": "harmonic_pinch", "harmonic": "harmonic",
        "slideUnpitchTo": "slide_unpitch_to", "slideTo": "slide_to",
        "linkNext": "link_next",
    }
    for xml_attr, py_attr in tech_attrs.items():
        if py_attr is None:
            feats.append(0.0)
            continue
        val = getattr(note, py_attr, False)
        feats.append(1.0 if val and val not in (False, 0, -1) else 0.0)

    return feats


def _encode_zone_notes(notes, chords) -> tuple:
    """Encode notes+chord_notes into (MAX_NOTES, NOTE_FEAT_DIM) + pad_mask."""
    all_notes = list(notes)
    for chord in chords:
        for cn in getattr(chord, "notes", []):
            all_notes.append(cn)
    all_notes = sorted(all_notes, key=lambda n: getattr(n, "time", 0))[:MAX_NOTES]
    n_real = len(all_notes)

    matrix = np.zeros((MAX_NOTES, NOTE_FEAT_DIM), dtype=np.float32)
    for i, note in enumerate(all_notes):
        matrix[i] = _encode_rs_note(note)

    pad_mask = np.ones(MAX_NOTES, dtype=bool)
    pad_mask[:n_real] = False
    if n_real == 0:
        pad_mask[0] = False

    return matrix, pad_mask


def _empty_transition() -> list:
    """Return a zero transition vector (32 features)."""
    return [0.0] * TRANS_FEAT_DIM


def _encode_transitions_for_zone(notes, chords, anchors, zone_start, zone_end) -> tuple:
    """
    Build transition features for notes in a zone.
    Simplified version of the extractor's build_transitions() for inference.
    """
    all_notes = list(notes)
    for chord in chords:
        for cn in getattr(chord, "notes", []):
            all_notes.append(cn)
    all_notes = sorted(all_notes, key=lambda n: getattr(n, "time", 0))[:MAX_NOTES]
    n_real = len(all_notes)

    matrix = np.zeros((MAX_NOTES, TRANS_FEAT_DIM), dtype=np.float32)

    for i, note in enumerate(all_notes):
        t    = getattr(note, "time", 0)
        fret = getattr(note, "fret", 0)

        # Previous note
        prev = all_notes[i-1] if i > 0 else None
        next_ = all_notes[i+1] if i < n_real - 1 else None

        prev_fret = getattr(prev, "fret", 0) if prev else 0
        next_fret = getattr(next_, "fret", 0) if next_ else 0
        fret_jump = abs(fret - prev_fret) if (fret > 0 and prev_fret > 0) else 0

        # Current anchor
        cur_anchor = None
        for a in anchors:
            if a.time <= t:
                cur_anchor = a
            else:
                break
        anchor_fret = cur_anchor.fret if cur_anchor else 1

        # Slide features
        slide_to  = getattr(note, "slide_to", -1)
        slide_up  = getattr(note, "slide_unpitch_to", -1)
        has_slide = slide_to > 0 or slide_up > 0
        slide_end = slide_to if slide_to > 0 else slide_up
        slide_delta = (slide_end - fret) if (has_slide and fret > 0 and slide_end > 0) else 0

        # HO/PO
        is_ho = getattr(note, "hammer_on", False)
        is_po = getattr(note, "pull_off", False)
        hopo_delta = (fret - prev_fret) if (is_ho or is_po) and prev else 0

        # HOFN
        is_hofn = is_ho and prev is None

        # Tap
        is_tap = getattr(note, "tap", False)

        tr = [
            # General (8)
            prev_fret / 22.0,
            getattr(prev, "string", 0) / 5.0 if prev else 0.0,
            min(t - getattr(prev, "time", t), 2.0) / 2.0 if prev else 0.0,
            next_fret / 22.0,
            getattr(next_, "string", 0) / 5.0 if next_ else 0.0,
            min(fret_jump, 22.0) / 22.0,
            1.0 if fret_jump > 4 else 0.0,
            0.0,  # string cross (simplified)

            # Anchor change (4)
            0.0,  # anchor_changes_here (not available at inference time per-note)
            anchor_fret / 22.0,
            anchor_fret / 22.0,
            0.0,

            # Slide (6)
            fret / 22.0 if has_slide else 0.0,
            max(0, slide_end) / 22.0 if has_slide else 0.0,
            max(-1.0, min(1.0, slide_delta / 11.0)),
            1.0 if slide_delta > 0 else 0.0,
            1.0 if (has_slide and slide_end > 0 and
                    not (anchor_fret <= slide_end <= anchor_fret + 4)) else 0.0,
            0.0,  # slide_anchor_change_follows (can't know at note level)

            # HO/PO (6)
            max(-1.0, min(1.0, hopo_delta / 4.0)),
            1.0 if hopo_delta > 0 else 0.0,
            1.0 if ((is_ho and hopo_delta > 0) or (is_po and hopo_delta < 0)) else 0.0,
            min(abs(hopo_delta), 8.0) / 8.0,
            1.0 if (anchor_fret <= fret <= anchor_fret + 4) else 0.0,
            0.0,

            # HOFN (4)
            1.0 if is_hofn else 0.0,
            0.0,  # anchor_established (unknown at inference)
            0.0,
            max(1, min(4, fret - anchor_fret + 1)) / 4.0 if (is_hofn and fret > 0) else 0.0,

            # Tap (4)
            fret / 22.0 if is_tap else 0.0,
            0.0,
            1.0 if (is_tap and not (anchor_fret <= fret <= anchor_fret + 4)) else 0.0,
            0.0,
        ]

        assert len(tr) == TRANS_FEAT_DIM
        matrix[i] = tr

    pad_mask = np.ones(MAX_NOTES, dtype=bool)
    pad_mask[:n_real] = False
    if n_real == 0:
        pad_mask[0] = False

    return matrix, pad_mask


def _encode_anchor_context(notes, chords, anchor_fret, anchor_duration,
                           avg_bpm, arrangement) -> list:
    """Build 10-feature context vector for anchor model v1."""
    frets = [getattr(n, "fret", 0) for n in notes if getattr(n, "fret", 0) > 0]
    fret_min = min(frets) if frets else 0
    fret_max = max(frets) if frets else 0
    return [
        min(len(notes), 64) / 64.0,
        min(len(chords), 32) / 32.0,
        fret_min / 22.0,
        fret_max / 22.0,
        (fret_max - fret_min) / 22.0,
        min(avg_bpm or 120, 300) / 300.0,
        min(anchor_duration, 10.0) / 10.0,
        1.0 if arrangement == "Lead"   else 0.0,
        1.0 if arrangement == "Bass"   else 0.0,
        1.0 if arrangement == "Rhythm" else 0.0,
    ]


def _encode_context(notes, chords, anchor_fret, anchor_duration,
                    avg_bpm, arrangement, prev_anchor, next_anchor) -> list:
    """Build 42-feature context vector for v3 fingering models."""
    frets = [getattr(n, "fret", 0) for n in notes if getattr(n, "fret", 0) > 0]
    fret_min = min(frets) if frets else 0
    fret_max = max(frets) if frets else 0
    note_count  = len(notes)
    chord_count = len(chords)

    NEIGHBOUR_DIM = 8

    def encode_neighbour(a_fret, a_width, n_count, c_count, f_min, f_max):
        return [
            a_fret / 22.0,
            a_width / 8.0,
            min(n_count, 64) / 64.0,
            min(c_count, 32) / 32.0,
            f_min / 22.0,
            f_max / 22.0,
            (f_max - f_min) / 22.0,
            0.0,
        ]

    base = [
        min(note_count, 64) / 64.0,
        min(chord_count, 32) / 32.0,
        fret_min / 22.0,
        fret_max / 22.0,
        (fret_max - fret_min) / 22.0,
        anchor_fret / 22.0,
        min(avg_bpm or 120, 300) / 300.0,
        min(anchor_duration, 10.0) / 10.0,
        1.0 if arrangement == "Lead"   else 0.0,
        1.0 if arrangement == "Bass"   else 0.0,
    ]

    if prev_anchor:
        base.extend(encode_neighbour(prev_anchor.fret, prev_anchor.width, 0, 0, 0, 0))
    else:
        base.extend([0.0] * NEIGHBOUR_DIM)

    base.extend([0.0] * NEIGHBOUR_DIM)

    if next_anchor:
        base.extend(encode_neighbour(next_anchor.fret, next_anchor.width, 0, 0, 0, 0))
    else:
        base.extend([0.0] * NEIGHBOUR_DIM)

    base.extend([0.0] * NEIGHBOUR_DIM)

    return base


# ─── Anchor Prediction ────────────────────────────────────────────────────────

def predict_anchors(rs_notes, rs_chords, naive_anchors, avg_bpm, arrangement, device, models):
    """
    Replace naive anchor list with ML-predicted anchors.

    Uses the same zone boundaries as the naive anchors (timing is preserved)
    but replaces the fret positions with model predictions.
    """
    anchor_model = models["anchor"]
    song_length  = max((getattr(n, "time", 0) + getattr(n, "sustain", 0)
                        for n in rs_notes), default=0) + 1.0

    if not naive_anchors:
        return naive_anchors

    # Build zone boundaries from naive anchors
    zone_ends = []
    for i in range(len(naive_anchors)):
        t_end = naive_anchors[i+1].time if i+1 < len(naive_anchors) else song_length
        zone_ends.append(t_end)

    improved = []

    for i, anchor in enumerate(naive_anchors):
        t_start = anchor.time
        t_end   = zone_ends[i]

        # Get notes in this zone
        zone_notes  = [n for n in rs_notes  if t_start <= getattr(n, "time", 0) < t_end]
        zone_chords = [c for c in rs_chords if t_start <= getattr(c, "time", 0) < t_end]

        if not zone_notes and not zone_chords:
            improved.append(anchor)
            continue

        # Encode
        note_matrix, pad_mask = _encode_zone_notes(zone_notes, zone_chords)

        prev_anchor = improved[-1] if improved else None
        next_anchor = naive_anchors[i+1] if i+1 < len(naive_anchors) else None
        duration    = t_end - t_start

        context = _encode_anchor_context(
            zone_notes, zone_chords, anchor.fret, duration,
            avg_bpm, arrangement
        )

        # Run model
        with torch.no_grad():
            notes_t   = torch.tensor(note_matrix, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t    = torch.tensor(pad_mask,    dtype=torch.bool).unsqueeze(0).to(device)
            context_t = torch.tensor(context,     dtype=torch.float32).unsqueeze(0).to(device)
            logits    = anchor_model(notes_t, mask_t, context_t)
            pred_fret = logits.argmax(-1).item()

        # Clamp to valid range
        pred_fret = max(1, min(pred_fret, 22))

        # Build improved anchor preserving original time and width
        from gp2rs import RsAnchor
        improved.append(RsAnchor(time=anchor.time, fret=pred_fret, width=4))

    log.info("Anchors: %d zones, naive frets %s → ML frets %s",
             len(naive_anchors),
             [a.fret for a in naive_anchors[:5]],
             [a.fret for a in improved[:5]])

    return improved


# ─── Fingering Prediction ─────────────────────────────────────────────────────

def predict_fingerings(chord_templates, rs_notes, rs_chords, ml_anchors,
                       avg_bpm, arrangement, device, models):
    """
    Fill in finger assignments on chord templates.
    Returns a new list of ChordTemplates with finger fields populated.
    """
    from gp2rs import ChordTemplate

    if not chord_templates:
        return chord_templates

    # Select model
    arr_lower = arrangement.lower()
    if "bass" in arr_lower:
        return chord_templates  # bass fingering handled by string selection model
    elif "rhythm" in arr_lower:
        model = models.get("rhythm", models.get("lead"))
    else:
        model = models["lead"]

    if model is None:
        return chord_templates

    song_length = max((getattr(n, "time", 0) + getattr(n, "sustain", 0)
                       for n in rs_notes), default=0) + 1.0

    # Build anchor lookup
    def anchor_at(t):
        result = ml_anchors[0] if ml_anchors else None
        for a in ml_anchors:
            if a.time <= t:
                result = a
            else:
                break
        return result

    # Find which chords use each template and get their zone context
    template_predictions = {}  # template_idx → predicted fingers

    for tmpl_idx, tmpl in enumerate(chord_templates):
        # Find chords using this template
        using_chords = [c for c in rs_chords
                        if getattr(c, "template_idx", -1) == tmpl_idx]

        if not using_chords:
            continue

        # Use first occurrence for context
        ref_chord = using_chords[0]
        t         = getattr(ref_chord, "time", 0)
        anchor    = anchor_at(t)

        # Build zone around this chord's anchor
        if anchor:
            next_anchor = None
            for a in ml_anchors:
                if a.time > anchor.time:
                    next_anchor = a
                    break
            t_end = next_anchor.time if next_anchor else song_length
        else:
            t_end = t + 2.0

        t_start     = anchor.time if anchor else max(0, t - 1.0)
        zone_notes  = [n for n in rs_notes  if t_start <= getattr(n, "time", 0) < t_end]
        zone_chords = [c for c in rs_chords if t_start <= getattr(c, "time", 0) < t_end]

        note_matrix, pad_mask = _encode_zone_notes(zone_notes, zone_chords)
        trans_matrix, trans_mask = _encode_transitions_for_zone(
            zone_notes, zone_chords, ml_anchors, t_start, t_end
        )

        prev_anchor = None
        for a in ml_anchors:
            if a.time < (anchor.time if anchor else t):
                prev_anchor = a
        next_anchor_ctx = None
        if anchor:
            for a in ml_anchors:
                if a.time > anchor.time:
                    next_anchor_ctx = a
                    break

        context = _encode_context(
            zone_notes, zone_chords,
            anchor.fret if anchor else 1,
            t_end - t_start, avg_bpm, arrangement,
            prev_anchor, next_anchor_ctx
        )

        with torch.no_grad():
            notes_t  = torch.tensor(note_matrix,  dtype=torch.float32).unsqueeze(0).to(device)
            mask_t   = torch.tensor(pad_mask,      dtype=torch.bool).unsqueeze(0).to(device)
            trans_t  = torch.tensor(trans_matrix,  dtype=torch.float32).unsqueeze(0).to(device)
            tmask_t  = torch.tensor(trans_mask,    dtype=torch.bool).unsqueeze(0).to(device)
            ctx_t    = torch.tensor(context,       dtype=torch.float32).unsqueeze(0).to(device)
            logits   = model(notes_t, mask_t, trans_t, tmask_t, ctx_t)  # (1, 6, FINGER_CLASSES)
            preds    = logits.argmax(-1).squeeze(0).tolist()             # [6]

        # Convert class indices back to finger assignments (-1..4)
        fingers = [p - FINGER_OFFSET for p in preds]
        template_predictions[tmpl_idx] = fingers

    # Build new chord templates with filled fingerings
    new_templates = []
    for i, tmpl in enumerate(chord_templates):
        fingers = template_predictions.get(i, tmpl.fingers)
        new_templates.append(ChordTemplate(
            name=tmpl.name,
            frets=tmpl.frets,
            fingers=fingers,
        ))

    assigned = sum(1 for i in template_predictions)
    log.info("Fingering: %d/%d templates assigned", assigned, len(chord_templates))

    return new_templates


# ─── Bass String Prediction ───────────────────────────────────────────────────

def predict_bass_strings(rs_notes, ml_anchors, avg_bpm, device, models):
    """
    For bass arrangements, predict the correct string for each note
    using the bass string selection model.

    Returns a new list of RsNote with string fields updated.
    """
    from gp2rs import RsNote

    if not rs_notes or "bass" not in models:
        return rs_notes

    bass_model  = models["bass"]
    song_length = max((getattr(n, "time", 0) + getattr(n, "sustain", 0)
                       for n in rs_notes), default=0) + 1.0

    # Build anchor lookup
    def anchor_at(t):
        result = ml_anchors[0] if ml_anchors else None
        for a in ml_anchors:
            if a.time <= t:
                result = a
            else:
                break
        return result

    # Process by anchor zone
    new_notes = list(rs_notes)  # copy
    note_idx  = {id(n): i for i, n in enumerate(rs_notes)}

    # Get unique anchor zones
    zone_starts = sorted(set(a.time for a in ml_anchors))

    for zi, t_start in enumerate(zone_starts):
        t_end = zone_starts[zi+1] if zi+1 < len(zone_starts) else song_length
        anchor = anchor_at(t_start + 0.001)

        zone_notes  = [n for n in rs_notes if t_start <= getattr(n, "time", 0) < t_end]
        zone_chords = []  # bass rarely has chords

        if not zone_notes:
            continue

        note_matrix, pad_mask = _encode_zone_notes(zone_notes, zone_chords)
        trans_matrix, trans_mask = _encode_transitions_for_zone(
            zone_notes, zone_chords, ml_anchors, t_start, t_end
        )

        prev_anchor = None
        for a in ml_anchors:
            if a.time < t_start:
                prev_anchor = a
        next_anchor = None
        for a in ml_anchors:
            if a.time > t_start:
                next_anchor = a
                break

        context = _encode_context(
            zone_notes, zone_chords,
            anchor.fret if anchor else 1,
            t_end - t_start, avg_bpm, "Bass",
            prev_anchor, next_anchor
        )

        with torch.no_grad():
            notes_t  = torch.tensor(note_matrix,  dtype=torch.float32).unsqueeze(0).to(device)
            mask_t   = torch.tensor(pad_mask,      dtype=torch.bool).unsqueeze(0).to(device)
            trans_t  = torch.tensor(trans_matrix,  dtype=torch.float32).unsqueeze(0).to(device)
            tmask_t  = torch.tensor(trans_mask,    dtype=torch.bool).unsqueeze(0).to(device)
            ctx_t    = torch.tensor(context,       dtype=torch.float32).unsqueeze(0).to(device)
            logits   = bass_model(notes_t, mask_t, trans_t, tmask_t, ctx_t)  # (1, S, 4)
            preds    = logits.argmax(-1).squeeze(0).tolist()                  # [MAX_NOTES]

        # Update string assignments for real notes
        for ni, note in enumerate(zone_notes[:MAX_NOTES]):
            pred_string = max(0, min(3, preds[ni]))
            orig_idx    = note_idx.get(id(note), -1)
            if orig_idx >= 0:
                orig = rs_notes[orig_idx]
                new_notes[orig_idx] = RsNote(
                    time=orig.time, string=pred_string, fret=orig.fret,
                    sustain=orig.sustain, bend=orig.bend,
                    slide_to=orig.slide_to, slide_unpitch_to=orig.slide_unpitch_to,
                    hammer_on=orig.hammer_on, pull_off=orig.pull_off,
                    harmonic=orig.harmonic, harmonic_pinch=orig.harmonic_pinch,
                    palm_mute=orig.palm_mute, mute=orig.mute,
                    vibrato=orig.vibrato, accent=orig.accent,
                    tremolo=orig.tremolo, tap=orig.tap, link_next=orig.link_next,
                )

    return new_notes


# ─── Main Conversion Function ─────────────────────────────────────────────────

def convert_gp_file(gp_path: str, output_dir: str = None,
                    track_indices: list = None,
                    audio_offset: float = 0.0) -> dict:
    """
    Convert a Guitar Pro file to both baseline and ML-enhanced feedBack XMLs.

    Args:
        gp_path:      Path to GP file (.gp3/.gp4/.gp5/.gpx etc.)
        output_dir:   Directory for output XMLs (default: same as GP file)
        track_indices: Which GP tracks to convert (None = auto-select)
        audio_offset: Seconds offset for audio sync

    Returns:
        Dict mapping arrangement name to {"baseline": path, "ml": path}
    """
    import guitarpro
    from gp2rs import (
        _build_tempo_map, _build_playback_schedule, _compute_tuning,
        _is_bass_track, _gp_string_to_rs, _tick_to_seconds,
        _tempo_at_tick, _duration_to_seconds, _extract_year,
        auto_select_tracks, is_drum_track, is_piano_track,
        convert_track, convert_drum_track, convert_piano_track,
        RsNote, RsChord, RsAnchor, ChordTemplate, RsBeat, RsSection,
        _build_xml,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_models(device)

    gp_path    = Path(gp_path)
    output_dir = Path(output_dir) if output_dir else gp_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_gp(path: Path) -> "guitarpro.Song":
        """
        Parse any Guitar Pro file with error recovery.
        GP3/4/5: parsed directly via PyGuitarPro.
        GPX/GP6/7/8: converted to GP5 via alphaTab (Node.js) then parsed.
        """
        import guitarpro.models as gpm

        def _patch_and_parse(p: Path):
            """Parse with ChordAlteration error recovery for malformed GP5 files."""
            _orig = gpm.ChordAlteration.__new__
            def _safe(cls, value):
                try:
                    return _orig(cls, value)
                except ValueError:
                    return _orig(cls, 0)
            gpm.ChordAlteration.__new__ = _safe
            try:
                return guitarpro.parse(str(p))
            finally:
                gpm.ChordAlteration.__new__ = _orig

        suffix = path.suffix.lower()

        # GP3/4/5 — parse directly
        if suffix in (".gp3", ".gp4", ".gp5"):
            try:
                return guitarpro.parse(str(path))
            except Exception:
                log.warning("Standard parse failed, retrying with error recovery...")
                return _patch_and_parse(path)

        # GPX/GP6/7/8 — supported natively in gp2rs.py (feedback#418)
        # Always use canonical gp2rs.py from feedback/feedback main branch.
        # PyGuitarPro does NOT support these formats.
        if suffix in (".gpx", ".gp"):
            raise RuntimeError(
                f"{path.suffix} format requires gp2rs.convert_file() from "
                f"https://github.com/feedback/feedback — not PyGuitarPro directly."
            )


        # Unknown format — try direct parse with recovery
        log.warning("Unknown GP format %s — attempting direct parse", suffix)
        return _patch_and_parse(path)

    song = _parse_gp(gp_path)

    if track_indices is None:
        track_indices, auto_names = auto_select_tracks(str(gp_path))
    else:
        auto_names = {}

    results = {}

    for track_idx in track_indices:
        track    = song.tracks[track_idx]
        arr_name = auto_names.get(track_idx, "Lead")

        log.info("Processing track %d: %s (%s)", track_idx, track.name, arr_name)

        # ── Step 1: Generate baseline XML via gp2rs.py ────────────────────────
        if is_drum_track(track):
            baseline_xml = convert_drum_track(song, track_idx, audio_offset, arr_name)
        elif is_piano_track(track):
            baseline_xml = convert_piano_track(song, track_idx, audio_offset, arr_name)
        else:
            baseline_xml = convert_track(song, track_idx, audio_offset, arr_name)

        # Save baseline
        safe_name    = re.sub(r'[^\w\-]', '_', gp_path.stem)
        baseline_out = output_dir / f"{safe_name}_{arr_name}_baseline.xml"
        baseline_out.write_text(baseline_xml, encoding="utf-8")
        log.info("Baseline XML: %s", baseline_out)

        # ── Step 2: Re-run conversion to get intermediate data ─────────────────
        # We need the internal data structures, not just the XML string.
        # Re-parse the arrangement data directly.
        if is_drum_track(track) or is_piano_track(track):
            # For drums/piano, ML enhancement not applicable — copy baseline
            ml_out = output_dir / f"{safe_name}_{arr_name}_ml.xml"
            ml_out.write_text(baseline_xml, encoding="utf-8")
            results[arr_name] = {"baseline": str(baseline_out), "ml": str(ml_out)}
            continue

        # Re-run gp2rs internals to get structured data
        tempo_map = _build_tempo_map(song)
        schedule  = _build_playback_schedule(song, tempo_map)
        headers   = song.measureHeaders
        num_strings = len(track.strings)
        is_bass   = _is_bass_track(track)
        tuning    = _compute_tuning(track)
        avg_bpm   = song.tempo

        # Collect beats
        beats = []
        for entry in schedule:
            mh = headers[entry.mh_index]
            beats.append(RsBeat(time=entry.output_start_secs + audio_offset, measure=mh.number))
            num_beats = mh.timeSignature.numerator
            for b in range(1, num_beats):
                from gp2rs import GP_TICKS_PER_QUARTER
                sub_tick = mh.start + b * GP_TICKS_PER_QUARTER
                sub_off  = _tick_to_seconds(sub_tick, tempo_map) - entry.mh_authored_start_secs
                beats.append(RsBeat(
                    time=entry.output_start_secs + sub_off + audio_offset, measure=-1
                ))
        beats.sort(key=lambda b: b.time)

        # Collect sections
        sections = []
        section_counts = {}
        for entry in schedule:
            mh = headers[entry.mh_index]
            if mh.marker and mh.marker.title:
                name = mh.marker.title.strip().lower().replace(" ", "")
                section_counts[name] = section_counts.get(name, 0) + 1
                sections.append(RsSection(
                    name=name,
                    time=entry.output_start_secs + audio_offset,
                    number=section_counts[name],
                ))
        if not sections:
            sections.append(RsSection(name="default", time=audio_offset, number=1))

        # Collect notes and chords (same logic as gp2rs.convert_track)
        import guitarpro as gp
        rs_notes        = []
        rs_chords       = []
        chord_templates = []
        chord_tmpl_map  = {}

        for entry in schedule:
            measure = track.measures[entry.mh_index]
            for voice in measure.voices:
                for beat in voice.beats:
                    if not beat.notes:
                        continue
                    authored_secs = _tick_to_seconds(beat.start, tempo_map)
                    t = (authored_secs - entry.mh_authored_start_secs) + entry.output_start_secs + audio_offset
                    tempo = _tempo_at_tick(beat.start, tempo_map)
                    dur   = _duration_to_seconds(beat.duration, tempo)

                    beat_notes = []
                    for note in beat.notes:
                        if note.type == gp.NoteType.rest:
                            continue
                        rs_str = _gp_string_to_rs(note.string, num_strings)
                        fret   = note.value
                        if note.type == gp.NoteType.dead:
                            fret = max(fret, 0)
                        rn = RsNote(
                            time=t, string=rs_str, fret=fret,
                            sustain=dur if dur > 0.2 else 0.0,
                            mute=note.type == gp.NoteType.dead,
                        )
                        eff = note.effect
                        if eff.bend and eff.bend.points:
                            rn.bend = max(p.value for p in eff.bend.points) / 100.0
                        if eff.hammer:
                            rn.hammer_on = True
                        if eff.slides:
                            for slide in eff.slides:
                                if slide in (gp.SlideType.shiftSlideTo, gp.SlideType.legatoSlideTo):
                                    rn.link_next = True
                        if eff.harmonic:
                            if isinstance(eff.harmonic, gp.PinchHarmonic):
                                rn.harmonic_pinch = True
                            else:
                                rn.harmonic = True
                        if eff.palmMute:      rn.palm_mute = True
                        if eff.accentuatedNote or eff.heavyAccentuatedNote: rn.accent = True
                        if eff.ghostNote:     rn.mute = True
                        if getattr(eff, "vibrato", False): rn.vibrato = True
                        if eff.tremoloPicking: rn.tremolo = True
                        beat_notes.append(rn)

                    if not beat_notes:
                        continue

                    if len(beat_notes) == 1:
                        rs_notes.append(beat_notes[0])
                    else:
                        used  = max((n.string for n in beat_notes if 0 <= n.string < num_strings), default=-1)
                        width = max(6, used + 1)
                        frets = [-1] * width
                        for n in beat_notes:
                            if 0 <= n.string < width:
                                frets[n.string] = n.fret
                        fret_key = tuple(frets)
                        if fret_key not in chord_tmpl_map:
                            chord_name = ""
                            if beat.effect and beat.effect.chord:
                                chord_name = beat.effect.chord.name or ""
                            idx = len(chord_templates)
                            chord_templates.append(ChordTemplate(
                                name=chord_name, frets=list(frets), fingers=[-1] * width,
                            ))
                            chord_tmpl_map[fret_key] = idx
                        rs_chords.append(RsChord(
                            time=t,
                            template_idx=chord_tmpl_map[fret_key],
                            notes=beat_notes,
                        ))

        rs_notes.sort(key=lambda n: n.time)
        rs_chords.sort(key=lambda c: c.time)

        # Compute naive anchors (same as gp2rs)
        all_timed_frets = [(n.time, n.fret) for n in rs_notes if n.fret > 0]
        for c in rs_chords:
            for cn in c.notes:
                if cn.fret > 0:
                    all_timed_frets.append((cn.time, cn.fret))
        all_timed_frets.sort()

        naive_anchors = []
        if all_timed_frets:
            naive_anchors.append(RsAnchor(time=audio_offset, fret=max(1, all_timed_frets[0][1] - 1), width=4))
            for t_f, fret in all_timed_frets:
                a_lo = naive_anchors[-1].fret
                a_hi = a_lo + naive_anchors[-1].width
                if fret < a_lo or fret > a_hi:
                    new_fret = max(1, fret - 1)
                    if new_fret != naive_anchors[-1].fret:
                        naive_anchors.append(RsAnchor(time=t_f, fret=new_fret, width=4))
        else:
            naive_anchors.append(RsAnchor(time=audio_offset, fret=1, width=4))

        # Song length
        if schedule:
            last = schedule[-1]
            song_length = last.output_start_secs + last.duration_secs + audio_offset
        else:
            song_length = audio_offset

        # ── Step 3: ML enhancement ────────────────────────────────────────────
        log.info("Running ML anchor prediction...")
        ml_anchors = predict_anchors(
            rs_notes, rs_chords, naive_anchors, avg_bpm, arr_name, device, models
        )

        log.info("Running ML fingering prediction...")
        ml_templates = predict_fingerings(
            chord_templates, rs_notes, rs_chords, ml_anchors,
            avg_bpm, arr_name, device, models
        )

        ml_notes = rs_notes
        if is_bass:
            log.info("Running ML bass string prediction...")
            ml_notes = predict_bass_strings(rs_notes, ml_anchors, avg_bpm, device, models)

        # ── Step 4: Build ML XML ──────────────────────────────────────────────
        ml_xml = _build_xml(
            title=song.title or gp_path.stem,
            artist=song.artist or "Unknown",
            album=song.album or "",
            year=_extract_year(song),
            arrangement=arr_name,
            tuning=tuning,
            num_strings=num_strings,
            song_length=song_length,
            audio_offset=audio_offset,
            beats=beats,
            sections=sections,
            notes=ml_notes,
            chords=rs_chords,
            chord_templates=ml_templates,
            anchors=ml_anchors,
            tempo=avg_bpm,
        )

        ml_out = output_dir / f"{safe_name}_{arr_name}_ml.xml"
        ml_out.write_text(ml_xml, encoding="utf-8")
        log.info("ML XML: %s", ml_out)

        results[arr_name] = {
            "baseline": str(baseline_out),
            "ml":       str(ml_out),
        }

    return results


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python inference_engine_v1.py <gp_file> [output_dir]")
        sys.exit(1)

    gp_path    = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Converting: {gp_path}")
    results = convert_gp_file(gp_path, output_dir)

    print("\nOutput files:")
    for arr_name, paths in results.items():
        print(f"\n  {arr_name}:")
        print(f"    Baseline: {paths['baseline']}")
        print(f"    ML:       {paths['ml']}")
