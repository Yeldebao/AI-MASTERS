"""Regenerate the paper figures from the executed notebook (run from this folder: python make_figures.py).

Vector plots are rebuilt from the per-epoch logs printed by the notebook; the interpretability panels are crops of
the figures embedded in the notebook outputs."""
import base64
import io
import json
import re

import nbformat
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

NOTEBOOK = "../Kvasir_CNN_vs_Transformer_V6.ipynb"
OUT = "figures"
nb = nbformat.read(NOTEBOOK, 4)

# Per-epoch histories from the training logs
pat = re.compile(r"^\[(.+?)\] epoch\s+(\d+)\s+train loss ([\d.]+)\s+val loss ([\d.]+)\s+val acc ([\d.]+)%\s+val macro-F1 ([\d.]+)")
H = {}
for cell in nb.cells:
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream" and o.get("name") == "stdout":
            for line in o.text.splitlines():
                m = pat.match(line)
                if m:
                    h = H.setdefault(m[1], {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []})
                    for k, v in zip(h, m.groups()[2:]):
                        h[k].append(float(v))


def notebook_image(marker):
    """First PNG output of the code cell whose source contains `marker`."""
    for cell in nb.cells:
        if cell.cell_type == "code" and marker in cell.source:
            for o in cell.outputs:
                if "data" in o and "image/png" in o.data:
                    return np.asarray(Image.open(io.BytesIO(base64.b64decode(o.data["image/png"]))).convert("RGB"))
    raise KeyError(marker)

C = {"ResNet-50": "#2a78d6", "EfficientNet-B0": "#eb6834", "ViT-S/16": "#1baf7a", "Hybrid R50+Transformer": "#4a3aa7", "Ensemble": "#52514e"}
LBL = {"ResNet-50": "ResNet-50", "EfficientNet-B0": "EfficientNet-B0", "ViT-S/16": "ViT-S/16", "Hybrid R50+Transformer": "Hybrid", "Ensemble": "Ensemble"}
BEST = {"ResNet-50": 12, "EfficientNet-B0": 15, "ViT-S/16": 19, "Hybrid R50+Transformer": 6}
MAIN = list(BEST)

# Convergence
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.55))
for m in MAIN:
    e = np.arange(1, len(H[m]["val_f1"]) + 1)
    a.plot(e, H[m]["val_loss"], color=C[m], label=LBL[m])
    b.plot(e, H[m]["val_f1"], color=C[m])
    b.plot(BEST[m], H[m]["val_f1"][BEST[m] - 1], "o", ms=4, color=C[m], mec="white", mew=0.8)
a.set(xlabel="epoch", ylabel="validation loss")
b.set(xlabel="epoch", ylabel="validation macro-F1", ylim=(0.77, 0.955))
for ax in (a, b):
    ax.grid(alpha=0.5, lw=0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
a.legend(frameon=False, loc="upper right", handlelength=1.2)
plt.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(f"{OUT}/convergence.pdf")

# Performance vs cost
pts = {  # GMACs, latency batch 1 (ms), F1, CI low, CI high
    "ViT-S/16": (4.5985, 7.196, 0.9358, 0.923, 0.949), "ResNet-50": (4.0872, 8.856, 0.9207, 0.906, 0.935),
    "EfficientNet-B0": (0.3845, 11.346, 0.9124, 0.897, 0.927), "Hybrid R50+Transformer": (4.8685, 10.360, 0.9062, 0.890, 0.923),
    "Ensemble": (9.467, 17.556, 0.9383, 0.925, 0.952)}
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.6), sharey=True)
for ax, k, xl in [(axes[0], 0, "GMACs per image"), (axes[1], 1, "latency, batch 1 (ms)")]:
    for m, v in pts.items():
        ax.errorbar(v[k], v[2], yerr=[[v[2] - v[3]], [v[4] - v[2]]], fmt="s" if m == "Ensemble" else "o", ms=4.5,
                    color=C[m], mec="white", mew=0.7, elinewidth=0.8, capsize=1.5)
    ax.set_xlabel(xl)
    ax.set_xlim(0, None)
    ax.grid(alpha=0.5, lw=0.5)
off = {"ViT-S/16": (-27, 3), "ResNet-50": (-33, -1), "EfficientNet-B0": (5, -2), "Hybrid R50+Transformer": (5, -6), "Ensemble": (-28, -9)}
for m, v in pts.items():
    axes[0].annotate(LBL[m], (v[0], v[2]), textcoords="offset points", xytext=off[m], fontsize=5.5, color="#3a3936")
axes[0].set_ylabel("test macro-F1")
plt.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(f"{OUT}/cost.pdf")

# Sensitivity
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.65))
e = np.arange(1, len(H["EfficientNet-B0"]["val_f1"]) + 1)
a.plot(e, H["EfficientNet-B0"]["val_f1"], color="#0b0b0b", lw=1.8, label="baseline")
for col, t in zip(["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
                  ["B0 · LR 1e-5", "B0 · LR 1e-3", "B0 · batch 128", "B0 · 64 px", "B0 · no augmentation", "B0 · from scratch"]):
    h = H[t]["val_f1"]
    a.plot(range(1, len(h) + 1), h, color=col, lw=1.1, label=t.replace("B0 · ", "").replace("no augmentation", "no aug."))
a.set(xlabel="epoch", ylabel="validation macro-F1", ylim=(0.3, 0.97), title="(a) EfficientNet-B0, one factor at a time")
a.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.0, columnspacing=0.6, fontsize=5.3)
for m, s, col in [("EfficientNet-B0", "B0 · from scratch", C["EfficientNet-B0"]), ("ViT-S/16", "ViT-S · from scratch", C["ViT-S/16"])]:
    b.plot(range(1, len(H[m]["val_f1"]) + 1), H[m]["val_f1"], color=col, label=f"{LBL[m]} pretrained")
    b.plot(range(1, len(H[s]["val_f1"]) + 1), H[s]["val_f1"], color=col, ls="--", label=f"{LBL[m]} from scratch")
b.set(xlabel="epoch", ylim=(0.3, 0.97), title="(b) value of pretraining")
b.legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=5.3)
for ax in (a, b):
    ax.grid(alpha=0.5, lw=0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
plt.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(f"{OUT}/sensitivity.pdf")

# Crops of the notebook figures
f6 = notebook_image('"fig6_interpretability"')
rows6 = [(176, 313), (314, 452), (453, 590), (731, 868), (870, 1007)]       # dyed-margin, esophagitis, cecum, z-line, polyp
Image.fromarray(np.concatenate([f6[21:45]] + [f6[a:b] for a, b in rows6])).save(f"{OUT}/interpretability.png")
f7 = notebook_image('"fig7_failure_cases"')
Image.fromarray(f7[0:313]).save(f"{OUT}/failure_cases.png")
Image.fromarray(notebook_image('"fig_scopeguide_bias"')).save(f"{OUT}/scopeguide.png")
Image.fromarray(notebook_image('"fig_dataset_examples"')).save(f"{OUT}/examples.png")
print("ok")
