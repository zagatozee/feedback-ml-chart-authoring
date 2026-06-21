# Slopsmith ML Chart Authoring — Claude Code Briefing Document
## Project: "Build Chart with Claude" Feature

---

## 1. Project Overview

This project integrates a trained ML authoring pipeline into the Slopsmith plugin editor as a "Build Chart with Claude" button. When a user imports a Guitar Pro file, this feature automatically assigns:

- **Anchor zones** — fret-hand position markers that minimise position shifts, derived from official chart authoring patterns
- **Chord fingerings** — per-string finger assignments (index/middle/ring/pinky) on every chord template
- **Bass string selection** — correct string chosen per note based on ergonomic position context
- **Technique mappings** — slides, HO/PO, HOFN, tapping, palm mute, vibrato, harmonics, bends

The goal is output indistinguishable from officially authored the source game/chart library content.

---

## 2. Repository Context

### Slopsmith repos involved:
- `slopsmith/slopsmith` — core backend (Flask/Python)
- `slopsmith/slopsmith-desktop` — Electron wrapper
- `slopsmith/slopsmith-plugin-editor` — the chart editor plugin (primary integration target)

### Key existing files (in `slopsmith-new/lib/`):
- `gp2rs.py` — GP file → feedBack XML converter (full-featured, handles GP3/4/5, repeat expansion, techniques, tuning, extended range). **This is the foundation — do not rewrite it.**
- `chart package.py` — chart package reader with AES decryption
- `song.py` — chart XML parser and song data models
- `cdlc_builder.py` — builds chart package from arranged XML data
- `gp2midi.py` — GP → MIDI conversion

### GPX/GP6/7/8 support:
GPX/GP6/GP7/GP8 support is built directly into `gp2rs.py` in the canonical Slopsmith repo (landed in slopsmith#418 by @zagatozee). **Always pull `gp2rs.py` fresh from `slopsmith/slopsmith` main branch** — do not use locally saved copies which may be outdated. The inference engine should import `gp2rs` directly from the installed Slopsmith lib.

GP8 additionally includes correct string order, bass naming, percussion decoding, and embedded audio extraction (slopsmith#714, #713). The `convert_gpx.js` alphaTab approach is **not needed** — disregard it.

---

## 3. ML Models

### Location
All models live at: `./\models\`

### Model inventory

| File | Architecture | Accuracy | Purpose |
|------|-------------|----------|---------|
| `anchor_model_v1.pt` | Single transformer encoder | 87.2% exact / 92.3% ±1 | Predicts anchor fret position per zone |
| `fingering_lead_v3.pt` | Dual encoder (note + transition) | 65.9% exact / ~91% per-string | Guitar Lead chord fingerings |
| `fingering_rhythm_v3.pt` | Dual encoder (note + transition) | 58.0% exact / ~87% per-string | Guitar Rhythm chord fingerings |
| `fingering_bass_v3.pt` | Dual encoder (note + transition) | 99.9% per-note | Bass string selection |

### Training data
- 560,999 anchor-zone records from 6,500+ Official chart library arrangements
- Stored at: `./\training_data_v3_*.jsonl`
- Source XMLs: `{YOUR_CHART_LIBRARY}\{Artist}\{Album}\{Song}\Official_*.xml`

### Model architecture specs

**Anchor model (v1 — single encoder):**
```python
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 3
D_FF      = 256
NOTE_FEAT = 20      # features per note
CONTEXT   = 10      # zone-level context features
MAX_NOTES = 32      # max notes per zone
OUTPUT    = 23      # fret classes (0-22)
```

**Fingering models (v3 — dual encoder):**
```python
# Main encoder (note sequence)
D_MODEL_MAIN  = 256, N_HEADS = 8, N_LAYERS = 4, D_FF = 512
# Transition encoder (technique transitions)
D_MODEL_TRANS = 128, N_HEADS = 4, N_LAYERS = 2, D_FF = 256
NOTE_FEAT     = 20      # same as anchor
TRANS_FEAT    = 32      # transition feature vector
CONTEXT       = 42      # 10 base + 4 neighbour zones × 8 features
OUTPUT_GUITAR = (6, 6)  # 6 strings × 6 finger classes
OUTPUT_BASS   = (32, 4) # 32 note positions × 4 string classes
FINGER_OFFSET = 1       # stored_finger = class_index - 1, so class 0 = finger -1 (unused)
```

### Reference implementations
- Full model class definitions: `inference_engine_v1.3.py` (in working dir)
- Feature encoding functions: `inference_engine_v1.3.py` → `_encode_rs_note()`, `_encode_anchor_context()`, `_encode_context()`, `_encode_transitions_for_zone()`
- Training scripts (for retraining): `train_models_v3.py`, `train_anchor_model_v1.py`
- Training data extractor (for v4+): `extract_training_data_v3.1.py`

---

## 4. Current Inference Engine

### File: `inference_engine_v1.3.py`
Working directory: `./\`

### What it does:
1. Parses a GP file via `guitarpro.parse()` (with error recovery for malformed chord data)
2. Runs `gp2rs.py`'s `convert_track()` to get the baseline XML and intermediate data structures (`rs_notes`, `rs_chords`, `chord_templates`, naive `anchors`)
3. Runs ML anchor prediction → replaces naive anchors with model predictions
4. Runs ML fingering prediction → fills in `fingers` fields on chord templates
5. For bass: runs ML string selection → updates `string` field on each note
6. Outputs two XMLs per arrangement: `_baseline.xml` (naive) and `_ml.xml` (ML-enhanced)

### Known issues to fix:
- GPX files fail because `guitarpro.parse()` doesn't support GP6+. Need to use the alphatab conversion path from `tab_import` plugin first
- The `auto_select_tracks()` call for GPX needs the converted GP5 data, not the raw GPX
- Error recovery for malformed GP5 chord data patches `ChordAlteration.__new__` — this should be made more robust

### Integration gap:
The inference engine currently runs as a standalone CLI tool. It needs to become a callable Python module that the editor plugin can invoke, returning structured data (not XML strings) that the editor can display and allow manual refinement before final export.

---

## 5. The "Build Chart with Claude" Feature Spec

### User flow:
1. User opens the Slopsmith editor
2. User imports a GP file (existing flow via tab_import plugin)
3. Editor shows track list with arrangement type assignments
4. User clicks **"Build Chart with Claude"** button
5. Progress indicator shows ("Analysing anchor positions...", "Assigning fingerings...", etc.)
6. Chart populates with ML-assigned anchors, fingerings, and technique markers
7. User can manually adjust anything in the editor as normal
8. User exports/saves as usual

### What the button should trigger:
```python
from inference_engine import enhance_arrangement

result = enhance_arrangement(
    rs_notes=notes,           # list of RsNote from gp2rs
    rs_chords=chords,         # list of RsChord from gp2rs
    chord_templates=templates, # list of ChordTemplate from gp2rs
    naive_anchors=anchors,    # list of RsAnchor from gp2rs
    arrangement_type="Lead",  # "Lead" | "Rhythm" | "Bass"
    avg_bpm=120.0,
)
# Returns:
# {
#   "anchors": [RsAnchor, ...],           # ML-predicted
#   "chord_templates": [ChordTemplate, ...], # with fingers filled in
#   "notes": [RsNote, ...],               # bass: with updated string assignments
# }
```

### Editor integration points:
- The editor likely has a JSON/IPC API between the Python backend and the Electron frontend
- The "Build Chart with Claude" action should be a new route in the editor plugin's `routes.py`
- The frontend button triggers this route and updates the chart state with the returned data
- Check existing routes in `plugins/editor/routes.py` to understand the data contract

---

## 6. Key Technical Decisions Already Made

### Tuning handling:
Tuning is **not** a model input feature. The anchor/fingering logic is purely geometric (fret numbers, string numbers, distances). Tuning is handled by `gp2rs.py` when computing fret positions from MIDI notes — by the time notes reach the ML models, they're already in fret/string space. This means one model set works across all tunings.

### Extended range instruments:
- 7/8 string guitar: supported at inference time via the same models — the additional strings extend the fretboard geometry without retraining
- 5/6 string bass: same approach — model trained on 4-string, extended range handled by path cost extension
- This is **not yet implemented** in the inference engine — needs a path solver layer

### Dynamic Difficulty (DD):
Not in scope for this tool. Existing community tools (e.g. DDC) handle DD generation from a complete single-difficulty chart. Our tool produces single-difficulty output; DD is applied afterward.

### No distribution of training data:
Model weights (.pt files) are distributable. The chart XML training data is not. Users bring their own GP files; the tool's output is their own work product.

---

## 7. What Needs Building (Priority Order)

### P0 — Core integration (required for "Build Chart with Claude" to work):
1. **Refactor inference engine into a proper module** — `enhance_arrangement()` function that takes structured data and returns structured data, no file I/O
2. **Editor plugin route** — new endpoint in `plugins/editor/routes.py` that calls the inference engine
3. **Frontend button** — "Build Chart with Claude" button in the editor UI that calls the route and updates chart state
4. **GPX support** — route the GPX → GP5 alphatab conversion before inference

### P1 — Quality improvements:
5. **Retrain anchor model with v3 data** — anchor_model_v1 was trained on v1 data (no transition features). Retraining on v3 data with the dual-encoder architecture should push accuracy from 87% → 91-93%
6. **Song-level anchor sequencing** — current model predicts each zone independently. A sequence model over the whole song would improve accuracy for position planning across long passages
7. **Extended range path solver** — explicit cost function for 7/8 string guitar and 5/6 string bass

### P2 — Feature completeness:
8. **Section/phrase detector** — when GP file has no section markers, detect verse/chorus/solo structure algorithmically and map to chart phrases
9. **Validation layer** — catch physically implausible note positions (unreachable stretches, wrong-direction HO/PO) before they reach the models
10. **Confidence scoring** — return a per-zone confidence score so the editor can highlight low-confidence anchor/fingering decisions for user review

### P3 — source data integration:
11. **Additional chart data** — Additional professionally authored chart sources can improve model accuracy. See training scripts for the expected XML format and folder structure.
12. **HumStrum data** — 13,000+ simplified chord charts available but tagged as `Official_HumStrum*`. Could augment rhythm training data if tagged with `is_humstrum: true` feature

---

## 8. Training Pipeline (for improvements)

### To retrain models:
```powershell
# Extract training data (v3 format with transition features)
python extract_training_data_v3.1.py

# Train all four models
python train_models_v3.py
```

### Training data extractor versions:
- `v1`: basic per-zone records, no transition features
- `v2`: adds neighbour zone context (±2 zones), better template selection
- `v3.1`: adds full transition features (slides, HO/PO, HOFN, tapping, general note-to-note)

### To add source data to training corpus:
1. Build RsCli: `cd {FEEDBACK_INSTALL_DIR}/rscli && dotnet build -c Release` (requires the SNG decompiler library)
2. Run `extract_legacy_xmls_v2.py` to extract XMLs from chart packages
3. Point extractor ROOT at `{YOUR_LEGACY_CHART_LIBRARY}\` in addition to chart library folder
4. Retrain

### v4 extractor improvements planned:
- Better HOFN detection across phrase boundaries
- Slide-to-anchor transition features (slide end fret → new anchor fret relationship)
- Per-string HO/PO direction validation against official data patterns

---

## 9. File Map

See the repository structure. Set `FEEDBACK_MODELS_DIR` and `FEEDBACK_LIB_DIR` environment variables to point at your local paths.


## 10. Accuracy Targets and Current State

| Component | Current | Target | Blocker |
|-----------|---------|--------|---------|
| Anchor placement | 87.2% exact | 91-93% | Retrain on v3 data with dual encoder |
| Lead fingering | 65.9% exact | 74-78% | Wider context + more epochs |
| Rhythm fingering | 58.0% exact | 65-72% | Data volume (only 37k records) |
| Bass string | 99.9% | 99.9% | Already at ceiling |
| GPX support | Broken | Working | Alphatab integration |
| Extended range | Not implemented | Working | Path solver layer |

---

## 11. Editor Updates Since Briefing Was Written

The following changes have landed in `slopsmith-plugin-editor` main that affect integration:

### v1.4.0 — New creation flow
- The toolbar *Create* button is now *New…* and opens a format picker: **Sloppak** or **chart package**
- New route: `POST /api/plugins/editor/create_sloppak` for audio+chart creation
- "Build Chart with Claude" should appear as an option **within** this creation flow, or as a post-import action button — not as a separate top-level entry point
- The legacy chart package path is unchanged

### v1.4.2 — Chord reconstruction
- `reconstructChords()` now correctly flattens arrangements before rebuilding chords from notes
- Our ML engine produces notes; the editor's `reconstructChords()` will rebuild chord objects from them — this is the correct integration pattern, don't try to inject chord objects directly

### Unreleased — No window.prompt()
- All `prompt()` calls replaced with `_editorPromptText` in-app modal
- Any UI added by this project (progress indicators, confirmation dialogs) **must use the same modal pattern**, not `window.prompt()` or `alert()`

### v1.2.0 — Tempo Map editor
- Users can now adjust the beat grid via a draggable tempo map UI
- Our anchor model uses timing derived from `gp2rs.py`'s tempo map at import time
- If a user adjusts the tempo map after ML authoring, note times shift but anchor times stay fixed — this is a known limitation to document, not fix immediately
- The "Build Chart with Claude" action should run **after** any tempo map adjustments, not before

### v1.3.0 — Drums first-class
- 18-piece drum vocabulary, drum import from GP/MIDI
- Our pipeline already handles drums as a pass-through (no ML enhancement for drums)
- Confirm the drum pass-through still works with the expanded piece vocabulary

---

## 12. Notes for Claude Code

- **Do not modify `gp2rs.py` directly** — it's a shared library used across multiple Slopsmith plugins. If you need different behaviour, wrap it.
- **The editor's data contract matters** — before writing the integration route, read `plugins/editor/routes.py` thoroughly to understand what data format the editor expects for notes, anchors, chord templates etc.
- **PyTorch models must load with `weights_only=True`** on PyTorch 2.x — this is already in the inference engine.
- **The UserWarning about `enable_nested_tensor`** is harmless — it's a PyTorch optimisation not available with `norm_first=True`. Suppress with `warnings.filterwarnings` if it clutters output.
- **Guitar Pro parse errors** — the `ChordAlteration` monkey-patch in `inference_engine_v1.3.py` handles malformed GP5 files. GPX needs the alphatab path. GP3/4 should work without any patching.
- **Model loading is slow** (~2-3 seconds) — load all four models once at server startup, not per-request. The inference engine's `_load_models()` function already caches them in a global dict.
- **GPU is available** — NVIDIA RTX 3060, CUDA 12.8, PyTorch 2.11+cu128. Models should run on GPU automatically via `torch.device("cuda")`.
- **BATCH_SIZE=1 at inference** — we process one song at a time, zone by zone. This is fine for interactive use; latency per song should be under 5 seconds on the 3060.

---

*Document version: v1.2 | Updated: GPX in gp2rs.py (slopsmith#418); editor v1.4.x changes documented*
*Working directory: ./\*
*Last updated: June 2026*
