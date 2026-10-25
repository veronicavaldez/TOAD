"""
export_predictions.py — nnU-Net prediction output -> docs/DATA_CONTRACT.md format.

Planned responsibilities:
  - Take nnU-Net's native nnUNetv2_predict output (run on imagesTs) and convert
    it to the exact docs/DATA_CONTRACT.md spec: one PNG per input image, same
    filename stem as the input, single-channel, 8-bit, pixel values = class
    IDs 0-4.
  - Since dataset.json's labels already map 1:1 to the contract, this should
    mostly be a format/naming pass-through — explicitly verify:
      * bit depth is 8-bit (not nnU-Net's sometimes-16-bit output)
      * single-channel (not an RGB visualization)
      * filenames don't carry extra nnU-Net-added suffixes that would break
        stem-matching against the input images

An HPG prediction was inspected before implementing load_prediction():
shape (192, 256), dtype uint8, and pixel values [0, 1, 2, 3, 4].
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for labels
from labels import CLASS_NAMES, validate_mask  # noqa: E402


def load_manifest(manifest_csv: str) -> list[dict]:
    """
    TODO: Read conversion_manifest.csv (written by convert_to_nnunet.py
    during data conversion — look at that file's `manifest_rows`/
    `write_dataset_json` section for the exact columns it writes:
    case_id, split, image_path, mask_path, stain).

    Return the rows where split == "test" as a list of dicts (csv.DictReader
    already gives you dicts — just filter and return them).
    """
    raise NotImplementedError


def original_stem(image_path: str) -> str:
    """
    TODO: Given a manifest row's original image_path (e.g.
    ".../Testing_images/Inputs/H&E/Raw256x192/Aging_H&E_016.tif"), return the
    filename stem the exported PNG should use, so it matches the original
    input image's name per DATA_CONTRACT.md ("same filename stem as the
    input image"). Path(...).stem is probably all you need here — but check
    what a real image_path from the manifest actually looks like first.
    """
    raise NotImplementedError


def load_prediction(pred_path: str) -> np.ndarray:
    """
    Load a single-channel 8-bit nnU-Net prediction containing contract class IDs.
    """
    mask = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read prediction file: {pred_path}")
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 2D prediction at {pred_path}; "
            f"got shape {mask.shape}"
        )
    if mask.dtype != np.uint8:
        raise ValueError(
            f"Expected an 8-bit prediction at {pred_path}; got dtype {mask.dtype}"
        )

    validate_mask(mask)
    return mask


def export_one(pred_path: str, out_path: str) -> None:
    """
    TODO:
      1. mask = load_prediction(pred_path)
      2. validate_mask(mask)  # from src.labels — raises if any pixel value
         isn't in CLASS_NAMES (0-4). If this raises, the prediction isn't
         already in contract label space and you'll need to add a remap
         step above before this point — don't silently clip/ignore invalid
         values.
      3. Make sure it's saved as single-channel 8-bit (cv2.imwrite with a
         2D uint8 array does this correctly).
      4. cv2.imwrite(out_path, mask)
    """
    raise NotImplementedError


def export_all(manifest_csv: str, predictions_dir: str, out_dir: str) -> None:
    """
    TODO:
      1. rows = load_manifest(manifest_csv)
      2. os.makedirs(out_dir, exist_ok=True)
      3. For each row:
         - predicted file is likely named f"{row['case_id']}.png" in
           predictions_dir — confirm by listing predictions_dir yourself
           rather than assuming.
         - out_name = original_stem(row["image_path"]) + ".png"
         - export_one(pred_path, os.path.join(out_dir, out_name))
      4. Print a summary: how many exported, and flag (without skipping)
         any manifest test row with no matching prediction file. That
         shouldn't happen, and if it does it's worth investigating why
         before treating this as done.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True,
                        help="conversion_manifest.csv from convert_to_nnunet.py")
    parser.add_argument("--predictions-dir", required=True,
                        help="Output directory from nnUNetv2_predict")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write DATA_CONTRACT.md-format PNGs")
    args = parser.parse_args()

    export_all(args.manifest_csv, args.predictions_dir, args.out_dir)


if __name__ == "__main__":
    main()
