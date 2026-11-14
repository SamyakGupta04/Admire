# Admire: Symmetry-Aware Image Inpainting with Stable Diffusion + LoRA

Fill missing regions of **faces** and **monuments** with a Stable Diffusion inpainting model that is
fine-tuned with LoRA. Monuments also get **Canny-edge ControlNet** guidance to keep geometry and symmetry.

| | Faces | Monuments |
|---|---|---|
| Base model | `stable-diffusion-v1-5/stable-diffusion-inpainting` | `sd2-community/stable-diffusion-2-inpainting` |
| Structure guidance | none | `thibaud/controlnet-sd21-canny-diffusers` (inference) |
| Data | CelebA aligned faces (15,000 train / 500 val / 500 test) | Indian Monuments, 24 monuments (3,372 train / 374 val / 1,148 test) |
| Resolution | 256 px | 512 px (256 px works for quick runs) |
| LoRA | r = 16, α = 32, dropout 0.05 on `to_q, to_k, to_v, to_out.0` | same |
| Eval masks | 5, 10, 20, 30, 40, 50 % | 3, 10, 15, 20 % |
| Metrics | PSNR, SSIM, masked PSNR, prompt OFF vs ON | same |

---

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the model weights (about 5.5 GB, fp16 safetensors, stored in `.hf_cache/` inside the repo):

```bash
python scripts/download_models.py          # or: faces | monuments
```

## 2. Data

Both datasets are public on Kaggle and download without an API key.

```bash
python scripts/prepare_data.py            # downloads + extracts + writes data/splits/*.txt
```

```
data/
├── celeba/img_align_celeba/*.jpg       202,599 faces
├── monuments/{train,test}/<monument>/   4,894 images, 24 monuments
└── splits/                             faces_{train,val,test}.txt, monuments_{train,val,test}.txt
```

Splits are fixed (seed 42) and committed, so results are reproducible. Images themselves are git-ignored.

## 3. Notebooks

Each notebook is self-contained. All settings are plain variables in the first code cell.
Open a notebook, select the `.venv` kernel, and run the cells from top to bottom.

| Notebook | Function | Output |
|---|---|---|
| `train_faces.ipynb` | Train LoRA on CelebA faces | `runs/faces/best`, `runs/faces/last`, `loss.png` |
| `train_monuments.ipynb` | Train LoRA on monuments | `runs/monuments/best`, `runs/monuments/last`, `loss.png` |
| `evaluate_faces.ipynb` | PSNR, SSIM, masked PSNR for faces, prompt off and on | `runs/faces/eval/results.csv` and example images |
| `evaluate_monuments.ipynb` | Same for monuments, with Canny ControlNet | `runs/monuments/eval/results.csv` and example images |

For a quick test, set `MAX_TRAIN = 40`, `MAX_VAL = 8`, `EPOCHS = 1` in a training notebook, and
`NUM_IMAGES = 2`, `STEPS = 10` in an evaluation notebook.

To evaluate the base model without fine-tuning, set `LORA_DIR = None`.

The notebooks use the GPU that is available (NVIDIA, or Apple Silicon, or the CPU).

## 4. Repository layout

```
train_faces.ipynb, train_monuments.ipynb          training
evaluate_faces.ipynb, evaluate_monuments.ipynb    evaluation
scripts/           prepare_data.py, download_models.py
data/splits/       fixed train / val / test image lists
```

## 5. Changes from the Kaggle scripts

The logic is the same (9-channel inpainting UNet, noise-prediction MSE, LoRA on attention, Canny ControlNet
for monuments). These problems were corrected:

1. **Local paths.** There are no `/kaggle/...` paths.
2. **Model ID.** `runwayml/stable-diffusion-inpainting` was removed from Hugging Face. The notebooks use the mirror `stable-diffusion-v1-5/stable-diffusion-inpainting`.
3. **Monument augmentation bug.** Random flip and colour jitter were applied *separately* to the image and the masked image, so the two often did not match. Augmentation now runs once, before masking.
4. **Train/inference mismatch.** Training filled the hole with black (−1). The diffusers pipelines fill it with 0 (mid-grey) at inference. Training now uses the pipeline convention.
5. **Memory.** The faces script cached 15,000 float images in RAM (about 12 GB). Images now load from disk.
6. **Checkpoints.** The notebooks save small LoRA adapters, not the full 3.4 GB UNet state dict.
7. **Evaluation scripts** referenced undefined names (`pd`, `SAVE_DIR`, `psnr_metric`). Test masks are now fixed per image (seeded), so prompt OFF and ON see the same masks.
8. **Masked PSNR** is reported for faces as well as monuments.
9. **Monument "3 %" masks** were two squares of 3 % *each* (about 6 % in total). They now share 3 % in total.

## 6. Notes for the report

- **Edge leakage (monuments).** The original evaluation computes the Canny map from the **undamaged**
  image, so ControlNet sees the structure inside the hole. Real damaged photos do not have that
  information. Report `EDGES_FROM = "original"` and `"masked"` side by side.
- **Full-image PSNR favours small masks.** Most pixels are untouched, so small masks score high.
  Quote masked PSNR when you compare mask sizes.
- **Symmetry.** The current pipeline gets symmetry only from Canny edges and prompts. An explicit
  symmetry loss on the predicted clean image, `λ·|x̂₀ − flip(x̂₀)|` inside the hole, is the natural next contribution.
- **Monument dataset.** It is a public substitute: Kaggle `danushkumarv/indian-monuments-image-dataset`.
  The test set is that dataset's own `test` folder.

## 7. Speed

On the development Mac, faces training at 256 px runs at about 1.3 images per second. One epoch of
15,000 images then takes about 3 to 4 hours. This is an estimate from a short test, not a measured full run.
An NVIDIA GPU (Kaggle or Colab) is much faster. The same notebooks run there without changes.
