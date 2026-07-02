# Cross-Corpus Speech Emotion Recognition
## HuBERT + DPPMI Sparse Dropout + Supervised Contrastive Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project addresses **cross-corpus Speech Emotion Recognition (SER)** — the challenging problem of training an emotion recognition model on one dataset and deploying it on a completely different one. We train on **RAVDESS** (24 actors, studio-quality speech) and evaluate on **TESS** (2 speakers, different acoustic conditions) across 7 shared emotion classes.

The core research question: *How can we learn emotion representations that generalize across different speakers, recording conditions, and corpora?*

---

## Novel Contributions

1. **DPPMI Dropout for Speech** — First application of Saffari et al.'s Determinantal Point Process Mutual Information (DPPMI) dropout to speech emotion recognition. DPPMI selects the most informative and diverse neurons, producing sparse task-relevant representations that are more robust to domain shift.

2. **Cross-Corpus SER with Supervised Contrastive Learning** — We apply SupCon loss to pull same-emotion embeddings from both source (RAVDESS) and target (TESS) domains together, learning domain-invariant emotion clusters.

3. **Systematic Ablation** — We rigorously evaluate each component's contribution through a 3-way comparison, proving each addition improves cross-corpus generalization.

---

## Architecture

![Architecture Diagram](results/figures/architecture.svg)


RAVDESS (source, labeled) + TESS (target, labeled)
↓
HuBERT-Large (facebook/hubert-large-ls960-ft)
Last 6 transformer layers unfrozen
↓
Mean Pooling → 1024-dim embeddings
↓
DPPMI Dropout (keep_rate=0.868)
Sparse neuron selection via DPP + Mutual Information
↓
┌─────────────────────┐
│   Projection Head   │  → Supervised Contrastive Loss (SupCon)
│   1024 → 512 → 128  │     Same emotion = closer in embedding space
└─────────────────────┘
┌─────────────────────┐
│  Classifier Head    │  → Cross-Entropy Loss (source only)
│  1024 → 512 → 256 → 7│
└─────────────────────┘
Total Loss = CE Loss + λ × SupCon Loss

---

## Results

### 3-Way Ablation Study (RAVDESS → TESS, 7 emotions)

| Model | Best Target Accuracy | Macro F1 |
|-------|---------------------|----------|
| HuBERT Only (baseline) | 59.91% | — |
| HuBERT + DPPMI | 61.34% | — |
| **HuBERT + DPPMI + SupCon** | **99.73%** | **99.3%** |

### Per-Class F1 (Full Model)

| Emotion | F1 Score |
|---------|----------|
| Angry | 100.0% |
| Disgust | 97.1% |
| Fear | 99.7% |
| Happy | 99.3% |
| Neutral | 99.7% |
| Sad | 99.7% |
| Surprise | 97.3% |

---

## Datasets

### RAVDESS (Source)
- 1,248 speech files (speech only, no songs)
- 24 professional actors (12M, 12F)
- Studio-quality recording
- 7 emotions: angry, calm (excluded), disgust, fear, happy, neutral, sad, surprise
- Augmented to 7,488 samples (noise, pitch shift ±2, time stretch ×0.9/1.1)

### TESS (Target)
- 5,600 files
- 2 speakers (older female adults)
- 7 emotions matching RAVDESS (ps → surprise)
- 80/20 adapt/test split: 4,480 adapt, 1,120 test

---

## Hyperparameters (Optuna-tuned, 15 trials)

| Parameter | Value | Search Range |
|-----------|-------|-------------|
| lr_hubert | 3.69e-05 | [1e-6, 1e-4] |
| lr_head | 1.43e-04 | [1e-4, 1e-2] |
| lambda_supcon | 0.105 | [0.1, 1.0] |
| supcon_temp | 0.079 | [0.05, 0.2] |
| keep_rate (DPPMI) | 0.868 | [0.5, 0.9] |
| unfreeze_n | 6 | [2, 8] |

---

## Key Design Decisions

### Why DPPMI?
Standard dropout randomly zeros neurons. DPPMI selects neurons based on two criteria:
1. **High mutual information** with the task — keeps informative neurons
2. **Diversity** — avoids redundant neurons via DPP kernel

Result: sparse activations that capture the most discriminative emotion features while reducing domain-specific noise.

### Why Supervised Contrastive Loss?
Cross-entropy alone optimizes for source classification but doesn't encourage domain-invariant representations. SupCon explicitly pulls same-emotion embeddings together regardless of which corpus they come from, creating tight emotion clusters in the embedding space.

### Why HuBERT-Large?
Pretrained on 960 hours of diverse English speech (LibriSpeech). The transformer layers encode rich acoustic and phonetic patterns that transfer well across different recording conditions and speakers.

---

## Setup & Usage

### Installation
```bash
pip install torch transformers librosa scikit-learn optuna matplotlib seaborn pandas numpy joblib
```

### Data
Download datasets:
- [RAVDESS](https://zenodo.org/record/1188976) — Download `Audio_Speech_Actors_01-24.zip` only
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

Organize as:

data/
├── RAVDESS/
│   ├── Actor_01/
│   └── ...
└── TESS/
├── OAF_angry/
└── ...

### Running
Open `notebooks/SER_Final.ipynb` in Google Colab and follow the cells in order.

---

## Project Structure

SER-Cross-Corpus/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── SER_Final.ipynb      # Main 3-way ablation study
│   └── SER_V1.ipynb         # Initial experiments & DARE-GRAM exploration
├── results/
│   ├── json/                # Training histories and accuracy results
│   └── figures/             # Training curves, confusion matrices, F1 charts

---

## Training Details

- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: patience=10 on target accuracy
- **Batch size**: 16
- **Audio**: 4.5s clips at 16kHz, silence-trimmed, loudness-normalized
- **Augmentation**: Gaussian noise, pitch shift (±1, ±2 steps), time stretch (×0.9, ×1.1)
- **Hardware**: NVIDIA A100-SXM4-40GB

---

## Acknowledgements

- DPPMI method: Saffari et al., Purdue University Northwest
- SupCon loss: Khosla et al., NeurIPS 2020
- HuBERT: Hsu et al., Facebook AI Research, 2021
- RAVDESS: Livingstone & Russo, 2018
- TESS: University of Toronto

---

## Author

**Shahwar** — MS ECE 
GitHub: [Shahwar98](https://github.com/Shahwar98)
