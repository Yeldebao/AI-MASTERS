# Kvasir v2 — CNNs vs. Transformers vs. a CNN-Transformer hybrid

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GITHUB_USER/GITHUB_REPO/blob/main/Kvasir_CNN_vs_Transformer.ipynb)

Deep Learning final project: comparative evaluation of deep-learning models for the classification of
gastrointestinal endoscopy images (**Kvasir v2**, 8 classes × 1,000 images), all trained under the same
experimental protocol.

| Family | Model | Pretrained weights (timm) |
|---|---|---|
| Classic CNN | ResNet-50 | `resnet50.tv2_in1k` |
| Efficient CNN | EfficientNet-B0 | `efficientnet_b0.ra_in1k` |
| Pure Transformer | ViT-S/16 | `vit_small_patch16_224.augreg_in21k_ft_in1k` |
| Hybrid (built here) | ResNet-50 stages 1–3 + 4 Transformer layers | ResNet part from `resnet50.tv2_in1k`, Transformer from scratch |
| Ensemble | Soft voting of the combination chosen on the validation set | — |

The notebook covers data download and preprocessing, the justification of every architecture and hyperparameter,
training, a one-factor-at-a-time hyperparameter sensitivity study, the ensemble, results with bootstrap confidence
intervals and McNemar tests, computational cost, convergence, interpretability (Grad-CAM and attention rollout),
a shortcut-learning test (the ScopeGuide box) and the critical analysis.

## How to run (Google Colab)

1. Click the **Open in Colab** badge above.
2. *Runtime → Change runtime type → **L4 GPU*** (a T4 also works, about 2× slower).
3. *Runtime → **Run all***, and authorize Google Drive when asked.
   - Drive stores the image cache, checkpoints and results in `MyDrive/kvasir_dl_final/`, so a disconnected
     session can resume: just *Run all* again and finished runs are loaded instead of retrained.
   - To run without Drive, set `USE_DRIVE = False` in section 1.
4. Duration: about **45–60 min on an L4** for the full run (4 models + 7 sensitivity runs).
   Set `SMOKE_TEST = True` in section 2 for a ~5 min check (20 images per class, 2 epochs).

The dataset (2.5 GB) is downloaded automatically from [simula.no](https://datasets.simula.no/kvasir/) to the Colab
VM; nothing has to be uploaded manually.

## How to run locally

```bash
pip install -r requirements.txt
jupyter notebook Kvasir_CNN_vs_Transformer.ipynb
```

A CUDA GPU is strongly recommended. If the dataset is already extracted somewhere, set the environment variable
`KVASIR_DIR` to the `kvasir-dataset-v2` folder to skip the download. Outputs are written to `./outputs/`.

## Experimental protocol (identical for every model)

| | |
|---|---|
| Split | stratified 70 / 15 / 15 (5,600 / 1,200 / 1,200), seed 42 |
| Input | resized to 256×256, random 224 crop (train) / center crop (val, test) |
| Augmentation | horizontal + vertical flips, rotation ±15°, brightness/contrast ±10 % (no hue changes: color is diagnostic) |
| Optimizer | AdamW, LR 1e-4 (10× for the hybrid's new layers), weight decay 0.05 |
| Schedule | 1 epoch linear warm-up + cosine decay, max 20 epochs, early stopping (patience 4) on validation macro-F1 |
| Loss | cross-entropy with label smoothing 0.1 |
| Precision | bf16 (L4/A100) or fp16 + GradScaler (T4), gradient clipping 1.0 |
| Metrics | accuracy, macro-F1 (primary), per-class F1, 95 % bootstrap CIs, McNemar tests, parameters, GMACs, training time, peak GPU memory, latency, epochs to converge |

## Outputs

`results/figures/` (PDF + PNG, 300 dpi), `results/tables/` (CSV + LaTeX) and `results/runs/` (per-run JSON with the
configuration, training history and predicted probabilities).

## Dataset and citation

K. Pogorelov et al., "Kvasir: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection,"
*Proc. 8th ACM Multimedia Systems Conference (MMSys)*, 2017. The dataset is distributed by Simula Research
Laboratory for research and educational use; it is not included in this repository.
