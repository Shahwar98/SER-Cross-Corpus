# Cross-Corpus Speech Emotion Recognition
## HuBERT + DPPMI Sparse Dropout + Supervised Contrastive Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project addresses **cross-corpus Speech Emotion Recognition (SER)**: training on one dataset (RAVDESS) and evaluating on a completely different one (TESS) across 7 shared emotion classes.

> **Setup (as implemented in `notebooks/SER_Final.ipynb`):** The classifier is trained on RAVDESS labels (cross-entropy + supervised contrastive loss on RAVDESS embeddings). TESS audio is passed through the model during training, but TESS labels do not enter any loss. TESS is split 80/20 (stratified, seed 42); the 20% split (1,120 clips) is the test set reported below.

---

## Novel Contributions

1. **DPPMI Dropout for Speech**: Applies Saffari et al.'s DPPMI dropout to SER. It keeps the most informative and diverse neurons (DPP + mutual-information scoring), giving sparse representations.

2. **Supervised Contrastive Learning for cross-corpus SER**: A SupCon loss on RAVDESS embeddings is added to cross-entropy.

3. **3-way ablation**: HuBERT only, HuBERT + DPPMI, and HuBERT + DPPMI + SupCon, trained with the same tuned hyperparameters.

---

## Architecture

![Architecture Diagram](results/figures/architecture.svg)

---

## Results

### Ablation Study (RAVDESS to TESS, 7 emotions)

| Model | Best-epoch target accuracy | Final-epoch accuracy |
|-------|---------------------------|----------------------|
| HuBERT Only | 59.91% | 55.18% |
| HuBERT + DPPMI | 61.34% | 60.98% |
| HuBERT + DPPMI + SupCon | **64.29%** | 63.66% |

Numbers are read from `results/json/hist_*.json` (single run, seed 42).

**How to read these results**
- The same 20% TESS split is used for evaluation *and* for checkpoint selection, early stopping, the LR scheduler, and the Optuna search. The best-epoch numbers are therefore optimistic, and the final-epoch column is the more conservative figure.
- With 1,120 test clips, one percentage point is about 11 clips. With a single seed, the gaps between rows (1.4 and 3.0 points) are suggestive, not conclusive.
- An earlier version of this README reported 99.73% for the full model. That figure came from an evaluation in which test data had been exposed during training, and it has been removed.
- `result_hubert_only.json` stores 56.61%, which comes from an earlier smoothed-accuracy metric. The table uses the raw best-epoch accuracy from the training history (59.91%).

### Per-class results (full model, final-epoch model, 63.66% accuracy)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Angry | 79.6% | 83.1% | 81.3% |
| Disgust | 48.8% | 91.9% | 63.8% |
| Fear | 91.9% | 78.1% | 84.5% |
| Happy | 62.2% | 35.0% | 44.8% |
| Neutral | 54.8% | 28.7% | 37.7% |
| Sad | 58.7% | 55.0% | 56.8% |
| Surprise | 61.5% | 73.8% | 67.0% |
| **Macro avg** | 65.4% | 63.7% | 62.3% |

Happy and neutral are the weakest classes, and many errors are predicted as disgust. See `results/figures/` for the confusion matrix, per-class F1 and precision/recall charts.

---

## Datasets

| | RAVDESS (Source) | TESS (Target) |
|---|---|---|
| Files | 1,248 (aug. to 7,488) | 5,600 |
| Speakers | 24 actors | 2 speakers |
| Labels used | CE + SupCon | None in training (evaluation only) |
| Test split | None | 20% held out (1,120 clips) |

**7 shared emotions:** angry, disgust, fear, happy, neutral, sad, surprise

---

## Hyperparameters (Optuna-tuned)

| Parameter | Value |
|-----------|-------|
| lr_hubert | 3.69e-05 |
| lr_head | 1.43e-04 |
| lambda_supcon | 0.105 |
| supcon_temp | 0.079 |
| keep_rate (DPPMI) | 0.868 |
| unfreeze_n | 6 |

---

## Setup

```bash
pip install torch transformers librosa scikit-learn optuna matplotlib seaborn pandas numpy joblib
```

Download datasets:
- [RAVDESS](https://zenodo.org/record/1188976): Audio_Speech_Actors_01-24.zip only
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

Open notebooks/SER_Final.ipynb in Google Colab with A100 GPU.

---

## Project Structure

- `notebooks/SER_Final.ipynb`: 3-way ablation study (main)
- `notebooks/SER_V1.ipynb`: WavLM + DPPMI + SimCLR experiment
- `results/json/`: Training histories and accuracy metrics
- `results/figures/`: Training curves, confusion matrix, per-class F1 and precision/recall charts for the final full-model run, per-run graph PDFs for the two baselines, and the architecture diagram

---

## Training Details

- **Hardware**: NVIDIA A100-SXM4-40GB
- **Optimizer**: AdamW with differential LR
- **Scheduler**: ReduceLROnPlateau (driven by target test accuracy)
- **Audio**: 4.5s, 16kHz, silence-trimmed, loudness-normalized
- **Augmentation**: Gaussian noise, pitch shift, time stretch
- **Epochs**: 50 max, early stopping patience=10 (on target test accuracy)
- **Seed**: 42, single run per model

---

## Acknowledgements

- DPPMI: Saffari et al., Purdue University Northwest
- SupCon: Khosla et al., NeurIPS 2020
- HuBERT: Hsu et al., Facebook AI Research, 2021
- RAVDESS: Livingstone & Russo, PLOS ONE 2018
- TESS: University of Toronto

---

## Author

**Shahwar**, MS ECE
GitHub: [Shahwar98](https://github.com/Shahwar98)
