#!/usr/bin/env python
"""Download (if needed) and prepare both datasets, then write split files.

    python scripts/prepare_data.py            # both datasets
    python scripts/prepare_data.py faces      # CelebA only
    python scripts/prepare_data.py monuments  # Indian monuments only

Layout after running:
    data/celeba/img_align_celeba/*.jpg          202,599 aligned 178x218 faces (CelebA)
    data/monuments/{train,test}/<monument>/*     24 Indian monuments (Kaggle: danushkumarv/indian-monuments-image-dataset)
    data/splits/*.txt                           one image path per line (relative to repo root)

Faces split: shuffle (seed 42) -> 15,000 train / 500 val / 500 test (same sizes as the original setup).
Monuments split: dataset 'train' folder -> 90 % train / 10 % val; dataset 'test' folder -> test.
Both datasets are public on Kaggle and download without an API key.
"""
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def write_split(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(str(p) for p in items) + "\n")

DATA = ROOT / "data"
SPLITS = DATA / "splits"
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/{}"
SOURCES = {
    "faces": ("jessicali9530/celeba-dataset", DATA / "downloads/celeba"),
    "monuments": ("danushkumarv/indian-monuments-image-dataset", DATA / "downloads/monuments"),
}
SEED = 42


def download(kind):
    slug, folder = SOURCES[kind]
    folder.mkdir(parents=True, exist_ok=True)
    zpath = folder / f"{kind}.zip"
    extracted = folder / "extracted"
    if extracted.exists() and any(extracted.iterdir()):
        return extracted
    if not zpath.exists() or not zipfile.is_zipfile(zpath):
        print(f"downloading {slug} ...")
        subprocess.run(["curl", "-L", "--fail", "-o", str(zpath), KAGGLE.format(slug)], check=True)
    print(f"extracting {zpath.name} ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(extracted)
    return extracted


def is_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def rel(paths):
    return [p.relative_to(ROOT).as_posix() for p in paths]


def prepare_faces(n_train=15000, n_val=500, n_test=500):
    target = DATA / "celeba/img_align_celeba"
    if not target.exists():
        ex = download("faces")
        src = next(p for p in ex.rglob("img_align_celeba") if p.is_dir() and any(p.glob("*.jpg")))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
        for csv in ex.rglob("*.csv"):           # attributes, landmarks, official partition
            shutil.move(str(csv), str(target.parent / csv.name))
    imgs = sorted(target.glob("*.jpg"))
    print(f"CelebA images: {len(imgs):,}")
    random.Random(SEED).shuffle(imgs)
    need = n_train + n_val + n_test
    chosen = [p for p in imgs[: need + 200] if is_ok(p)][:need]
    write_split(SPLITS / "faces_train.txt", rel(chosen[:n_train]))
    write_split(SPLITS / "faces_val.txt", rel(chosen[n_train:n_train + n_val]))
    write_split(SPLITS / "faces_test.txt", rel(chosen[n_train + n_val:]))
    print(f"faces split -> train {n_train:,} / val {n_val} / test {n_test}")


def prepare_monuments(val_frac=0.1):
    target = DATA / "monuments"
    if not target.exists():
        ex = download("monuments")
        src = next(p for p in ex.rglob("images") if (p / "train").is_dir())
        shutil.move(str(src), str(target))
    for f in target.rglob("*"):                 # drop stray non-image files (e.g. a saved .html page)
        if f.is_file() and f.suffix.lower() not in {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}:
            f.unlink()
    train_all = [p for p in list_images(target / "train") if is_ok(p)]
    test = [p for p in list_images(target / "test") if is_ok(p)]
    random.Random(SEED).shuffle(train_all)
    n_val = int(len(train_all) * val_frac)
    write_split(SPLITS / "monuments_train.txt", rel(train_all[n_val:]))
    write_split(SPLITS / "monuments_val.txt", rel(train_all[:n_val]))
    write_split(SPLITS / "monuments_test.txt", rel(test))
    classes = sorted({p.parent.name for p in train_all})
    print(f"monuments: {len(classes)} monuments -> train {len(train_all) - n_val:,} / val {n_val} / test {len(test)}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["faces", "monuments"]
    if "monuments" in which:
        prepare_monuments()
    if "faces" in which:
        prepare_faces()
