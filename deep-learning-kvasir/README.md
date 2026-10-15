# Kvasir v6 — CNNs vs. Transformers vs. a CNN-Transformer hybrid

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yeldebao/AI-MASTERS/blob/main/deep-learning-kvasir/Kvasir_CNN_vs_Transformer_V6.ipynb)

Deep Learning final project: comparative evaluation of deep-learning models for the classification of
gastrointestinal endoscopy images (**Kvasir v6**, 8 classes × 1,000 images), all trained under the same
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

## Results (test set, 1,200 images, NVIDIA L4)

| Model | Accuracy (%) | Macro-F1 [95 % CI] | Params (M) | GMACs | Train time (s) | Latency (ms, batch 1) |
|---|---|---|---|---|---|---|
| Ensemble ViT-S + Hybrid | 93.83 | 0.938 [0.925, 0.952] | 37.8 | 9.47 | 353 | 17.6 |
| **ViT-S/16** | **93.58** | **0.936 [0.923, 0.949]** | 21.7 | 4.60 | 190 | **7.2** |
| ResNet-50 | 92.08 | 0.921 [0.906, 0.935] | 23.5 | 4.09 | 246 | 8.9 |
| EfficientNet-B0 | 91.25 | 0.912 [0.897, 0.927] | **4.0** | **0.38** | 210 | 11.3 |
| Hybrid R50+Transformer | 90.67 | 0.906 [0.890, 0.923] | 16.1 | 4.87 | 162 | 10.4 |

Key findings (details in sections 14–15 of the notebook):
* ViT-S/16 is significantly better than the other three models (McNemar p < 0.05); the ensemble's +0.003 over it is not significant.
* Pretraining matters far more than architecture: training from scratch costs −0.19 (ViT-S) to −0.26 (EfficientNet-B0) macro-F1.
* EfficientNet-B0 reaches 97 % of the best macro-F1 with 1/12 of the compute, but has the highest single-image GPU latency.
* The main error for every model is esophagitis ↔ normal z-line, a fine-grained distinction.
* All models partly rely on a dataset shortcut (the ScopeGuide box, present only in colonoscopies): masking it costs 0.014–0.034 macro-F1.

## Deliverables

| | |
|---|---|
| Notebook (executed on an NVIDIA L4) | [`Kvasir_CNN_vs_Transformer_V6.ipynb`](Kvasir_CNN_vs_Transformer_V6.ipynb) |
| IEEE conference paper | [`paper/paper.pdf`](paper/paper.pdf) (LaTeX source and figures in `paper/`) |
| Presentation (Spanish, 12 slides) | [`presentation/presentacion_kvasir.pdf`](presentation/presentacion_kvasir.pdf), speaker notes in [`presentation/notas_del_orador.md`](presentation/notas_del_orador.md) |

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
cd deep-learning-kvasir && jupyter notebook Kvasir_CNN_vs_Transformer_V6.ipynb
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
