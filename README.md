# feedBack ML Chart Authoring

An ML-powered chart authoring engine for [feedBack](https://github.com/slopsmith/slopsmith-desktop) that converts Guitar Pro files into professionally-authored arrangements — with intelligent anchor zones, chord fingerings, and bass string selection.

## What It Does

When you import a Guitar Pro file, this tool automatically assigns:

- **Anchor zones** — fret-hand position markers that minimise unnecessary position shifts, derived from patterns in thousands of professionally authored charts
- **Chord fingerings** — per-string finger assignments (index/middle/ring/pinky) on every chord shape
- **Bass string selection** — ergonomic string choice per note based on position context
- **Technique mappings** — slides, hammer-ons/pull-offs, palm muting, vibrato, harmonics, bends, tapping

The goal is output that plays and feels like a professionally authored chart, not a raw tab-to-game conversion.

## How It Works

The tool uses four trained transformer models:

| Model | Purpose | Accuracy |
|-------|---------|----------|
| Anchor placement | Predicts fret-hand position per zone | 87.2% exact / 92.3% ±1 fret |
| Lead fingering | Chord finger assignments (guitar lead) | ~91% per-string |
| Rhythm fingering | Chord finger assignments (guitar rhythm) | ~87% per-string |
| Bass string | String selection per note | 99.9% |

Models were trained on anchor-zone records extracted from a large corpus of professionally authored charts, learning the ergonomic and stylistic patterns that make charts feel natural to play.

## Requirements

- Python 3.11+
- PyTorch 2.x with CUDA (GPU recommended, CPU works but slower)
- feedBack installed and running
- Guitar Pro files in GP3/GP4/GP5 format (GPX/GP6/GP7/GP8 via feedBack's built-in converter)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyguitarpro numpy
```

## Setup

**1. Clone this repo**
```bash
git clone https://github.com/zagatozee/feedback-ml-authoring
cd feedback-ml-authoring
```

**2. Download the model weights**

Download from [Releases](../../releases) and place in `models/`:
```
models/
  anchor_model_v1.pt
  fingering_lead_v3.pt
  fingering_rhythm_v3.pt
  fingering_bass_v3.pt
```

**3. Copy feedBack's lib folder**

The inference engine needs `gp2rs.py` from your feedBack installation:
```bash
# Pull direct from feedBack repo (always use latest)
git clone https://github.com/slopsmith/slopsmith lib_source
cp -r lib_source/lib ./lib
rm -rf lib_source
```

**4. Set environment variables (optional)**
```bash
# If your models or lib are in non-default locations
export FEEDBACK_MODELS_DIR=/path/to/models
export FEEDBACK_LIB_DIR=/path/to/lib
```

## Usage

### A/B Comparison (CLI)

Produces both a baseline and ML-enhanced XML for comparison:

```bash
python inference_engine_v1.py "path/to/song.gp5" "output/"
```

Output:
```
output/
  song_Lead_baseline.xml    ← standard gp2rs.py conversion
  song_Lead_ml.xml          ← ML-enhanced version
  song_Bass_baseline.xml
  song_Bass_ml.xml
```

Load both into feedBack's editor to compare anchor placement and fingerings side by side.

### As a Module

```python
from inference_engine_v1 import convert_gp_file

results = convert_gp_file(
    gp_path="song.gp5",
    output_dir="output/",
    audio_offset=0.0,
)

for arrangement, paths in results.items():
    print(f"{arrangement}:")
    print(f"  Baseline: {paths['baseline']}")
    print(f"  ML:       {paths['ml']}")
```

## feedBack Editor Integration

This tool is designed to integrate into feedBack's editor as a **"Build Chart with AI"** button. See [`claude_code_briefing_v1.3.md`](claude_code_briefing_v1.3.md) for the full integration spec.

The integration adds a new route to the editor plugin that calls the inference engine and populates the chart directly — no manual file handling required.

## Training Your Own Models

The model architecture and training scripts are included. To train on your own chart library:

**1. Extract training data**

Point `extract_training_data_v3.py` at a folder of arrangement XMLs in the standard format:
```
{ROOT}/{Artist}/{Album}/{Song}/{arrangement}.xml
```

Arrangement files should be named to indicate type (Lead/Rhythm/Bass).

```bash
# Edit ROOT in extract_training_data_v3.py to point at your chart library
python extract_training_data_v3.py
```

**2. Train models**

```bash
python train_models_v3_public.py
```

Training time on an NVIDIA RTX 3060: ~12-16 hours for all four models.

**3. Use your models**

Set `FEEDBACK_MODELS_DIR` to point at your newly trained models.

## Supported Formats

| Format | Support |
|--------|---------|
| GP3, GP4, GP5 | ✅ Native via PyGuitarPro |
| GPX / GP6 / GP7 / GP8 | ✅ Via feedBack's built-in converter |
| 6-string guitar | ✅ Full support |
| 7/8-string guitar | ✅ Inference-time extension |
| 4-string bass | ✅ Full support |
| 5/6-string bass | ✅ Inference-time extension |
| Drums | ✅ Pass-through (no ML enhancement) |
| Keys/Piano | ✅ Pass-through (no ML enhancement) |

## Known Limitations

- GPX files require feedBack's lib to be available (see Setup step 3)
- Rhythm fingering accuracy (87% per-string) is lower than Lead due to smaller training corpus — this will improve with more training data
- Anchor accuracy improves significantly when tempo map is correct — run feedBack's Tempo Map editor before using this tool if the GP file has timing issues
- Dynamic Difficulty (DD) is not generated — use DDC or similar tools afterward

## Contributing

PRs welcome. Key areas for improvement:

- **v4 training extractor** — slide→anchor transition features, better HOFN detection
- **Song-level anchor sequencing** — current model predicts zones independently; a sequence model would improve long-passage planning
- **Extended range path solver** — explicit cost function for 7/8 string instruments
- **Section detector** — algorithmic verse/chorus detection when GP file has no markers

See [`v3_extractor_spec.md`](v3_extractor_spec.md) for the detailed improvement spec.

## Licence

MIT — see [LICENSE](LICENSE).

Model weights are derived from training on licensed chart data and are provided for personal, non-commercial use.
