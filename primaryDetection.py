import cv2
import numpy as np
import random
import math
import json
import os
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================
IMAGE_FOLDER = "ellipses"          # folder with id_0.png, id_1.png, ...
ANNOTATIONS_FILE = "annotations.json"
WORLD_SPACE_SIZE = 100             # world coords (0..100)

# --- RHT PARAMETERS (tuned for small images) ---
RHT_EPOCHS = 3000          # number of random 5-point samples per pass
ACCUMULATOR_BIN_SIZE = 2   # bin size in pixels
MIN_VOTES = 3              # min bin count to accept ellipse
INLIER_THRESHOLD = 2.5     # how tight points must lie on ellipse

# --- SIMPLE PRIORS ON CENTER POSITION (world units) ---
PMMA_X_MIN = 30.0
PMMA_X_MAX = 70.0
PMMA_Y_MIN = 30.0
PMMA_Y_MAX = 70.0


# =========================================================
# Utility functions
# =========================================================
def get_scale_factor(img_width: int) -> float:
    """Map 0..img_width pixels -> 0..WORLD_SPACE_SIZE world units."""
    return img_width / WORLD_SPACE_SIZE


def convert_to_world(pixel_ellipse, scale: float):
    """
    Convert cv2.fitEllipse result (pixel coords) to world coords.

    pixel_ellipse: ((cx, cy), (d1, d2), angle_deg)
    """
    (cx, cy), (d1, d2), angle_deg = pixel_ellipse

    wx = cx / scale
    wy = cy / scale

    semi_axis_a = (d1 / 2.0) / scale
    semi_axis_b = (d2 / 2.0) / scale

    return {
        "center_x": wx,
        "center_y": wy,
        "semi_major_axis": max(semi_axis_a, semi_axis_b),
        "semi_minor_axis": min(semi_axis_a, semi_axis_b),
        "orientation_angle_rad": angle_deg * (math.pi / 180.0),
    }


def ellipse_ok_in_world(world_ellipse) -> bool:
    """
    Keep only ellipses whose center lies in the PMMA window.
    You can disable this prior by 'return True' if you want.
    """
    cx = world_ellipse["center_x"]
    cy = world_ellipse["center_y"]

    if not (PMMA_X_MIN <= cx <= PMMA_X_MAX):
        return False
    if not (PMMA_Y_MIN <= cy <= PMMA_Y_MAX):
        return False

    return True


def extract_edge_points(img, intensity_thresh: int = 40, blur_ksize: int = 3):
    """
    Extract edge points for ellipse fitting from a 50x50 heatmap-like image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # remove very dark background
    _, mask = cv2.threshold(gray, intensity_thresh, 255, cv2.THRESH_TOZERO)

    edges = cv2.Canny(mask, 50, 150)

    ys, xs = np.where(edges > 0)
    points = list(zip(xs.tolist(), ys.tolist()))
    return points


# =========================================================
# Randomized Hough Transform for ellipses
# =========================================================
def rht_single_pass(points, width, height):
    """
    One RHT pass: pick 5 random points repeatedly, fit ellipses,
    accumulate in quantized parameter space, return best ellipse & inliers.
    """
    if len(points) < 5:
        return None, []

    accumulator = defaultdict(int)
    bin_to_params = defaultdict(list)

    for _ in range(RHT_EPOCHS):
        sample = random.sample(points, 5)

        try:
            cand = cv2.fitEllipse(np.array(sample, dtype=np.int32))
            (cx, cy), (d1, d2), angle = cand

            if (
                np.isnan(cx)
                or np.isnan(cy)
                or np.isnan(d1)
                or np.isnan(d2)
            ):
                continue

            # basic sanity checks
            if not (-width < cx < 2 * width) or not (-height < cy < 2 * height):
                continue
            if d1 < 2 or d2 < 2:
                continue

            # normalise so d1 = semi-major axis
            if d1 < d2:
                d1, d2 = d2, d1
                angle = (angle + 90) % 180

            key = (
                int(cx / ACCUMULATOR_BIN_SIZE),
                int(cy / ACCUMULATOR_BIN_SIZE),
                int(d1 / ACCUMULATOR_BIN_SIZE),
                int(d2 / ACCUMULATOR_BIN_SIZE),
                int(angle / 10),
            )

            accumulator[key] += 1
            bin_to_params[key].append(((cx, cy), (d1, d2), angle))

        except Exception:
            continue

    if not accumulator:
        return None, []

    best_key = max(accumulator, key=accumulator.get)

    if accumulator[best_key] < MIN_VOTES:
        return None, []

    cands = bin_to_params[best_key]

    avg_cx = np.mean([c[0][0] for c in cands])
    avg_cy = np.mean([c[0][1] for c in cands])
    avg_d1 = np.mean([c[1][0] for c in cands])
    avg_d2 = np.mean([c[1][1] for c in cands])
    avg_ang = np.mean([c[2] for c in cands])

    final_ellipse = ((avg_cx, avg_cy), (avg_d1, avg_d2), avg_ang)

    # inliers (not really needed here, but kept for consistency)
    inlier_indices = []
    ang_rad = avg_ang * math.pi / 180.0
    cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)
    a, b = avg_d1 / 2.0, avg_d2 / 2.0

    if a < 0.1 or b < 0.1:
        return None, []

    for idx, (px, py) in enumerate(points):
        tx, ty = px - avg_cx, py - avg_cy
        xr = tx * cos_a + ty * sin_a
        yr = -tx * sin_a + ty * cos_a
        dist = (xr / a) ** 2 + (yr / b) ** 2

        if abs(dist - 1.0) < (INLIER_THRESHOLD / min(a, b)):
            inlier_indices.append(idx)

    return final_ellipse, inlier_indices


# =========================================================
# Per-image processing (PRIMARY only, at most 1 detection)
# =========================================================
def process_image_primary(img_path, gt_primary_ellipses):
    """
    Run detection on one image and return:
      - gt_count  = number of GT PRIMARY ellipses
      - det_count = 0 or 1 (we try to detect ONE primary ellipse)
    """
    img = cv2.imread(img_path)
    if img is None:
        return len(gt_primary_ellipses), 0

    h, w = img.shape[:2]
    scale = get_scale_factor(w)

    # 1) edge points
    all_points = extract_edge_points(img, intensity_thresh=40)

    detected_primary = None

    # 2) RHT: single pass (we just want the best ellipse)
    if len(all_points) >= 5:
        ellipse, _ = rht_single_pass(all_points, w, h)
        if ellipse is not None:
            world_ellipse = convert_to_world(ellipse, scale)
            if ellipse_ok_in_world(world_ellipse):
                detected_primary = world_ellipse

    # 3) Fallback if RHT failed
    if (detected_primary is None) and len(all_points) >= 5:
        try:
            pts = np.array([[x, y] for (x, y) in all_points], dtype=np.int32)
            fallback = cv2.fitEllipse(pts)
            world_fb = convert_to_world(fallback, scale)
            if ellipse_ok_in_world(world_fb):
                detected_primary = world_fb
        except Exception:
            pass

    gt_count = len(gt_primary_ellipses)
    det_count = 1 if detected_primary is not None else 0
    return gt_count, det_count


# =========================================================
# Main evaluation loop (PRIMARY-only)
# =========================================================
def main():
    print(f"--- STARTING RHT (PRIMARY ELLIPSE ONLY) ---")
    print(
        f"Bins: {ACCUMULATOR_BIN_SIZE} | Epochs: {RHT_EPOCHS} | Min Votes: {MIN_VOTES}"
    )

    try:
        with open(ANNOTATIONS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        print("JSON not found.")
        return

    processed = 0

    total_gt = 0
    total_det = 0
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    exact_match_count = 0

    # group annotations by image id, KEEP ONLY PRIMARY ellipses
    anns_by_id = {}
    for ann in data["annotations"]:
        prim = [e for e in ann["ellipses"] if e.get("type") == "primary"]
        anns_by_id[ann["image_id"]] = prim

    for entry in data["images"]:
        img_id = entry["id"]
        path = os.path.join(IMAGE_FOLDER, f"id_{img_id}.png")
        if not os.path.exists(path):
            path = os.path.join(IMAGE_FOLDER, f"image_{img_id}.png")
        if not os.path.exists(path):
            continue

        gt_prim = anns_by_id.get(img_id, [])
        gt_count, det_count = process_image_primary(path, gt_prim)

        processed += 1
        total_gt += gt_count
        total_det += det_count

        diff = det_count - gt_count
        abs_err_sum += abs(diff)
        sq_err_sum += diff * diff
        if det_count == gt_count:
            exact_match_count += 1

        if processed % 50 == 0:
            print(
                f"Processed {processed} images... (exact matches so far: {exact_match_count})"
            )

    if processed == 0:
        return

    mean_abs_error = abs_err_sum / processed
    rmse = math.sqrt(sq_err_sum / processed)
    mean_bias = (total_det - total_gt) / processed
    exact_accuracy = exact_match_count / processed

    print("\n" + "=" * 30)
    print("FINAL COUNT-BASED RESULTS (PRIMARY ONLY)")
    print("=" * 30)
    print(f"Images:                     {processed}")
    print(f"Total GT PRIMARY ellipses:  {total_gt}")
    print(f"Total detected ellipses:    {total_det}")
    print("-" * 30)
    print(f"Exact Count Accuracy:       {exact_accuracy:.2%}")
    print(f"Mean Absolute Count Error:  {mean_abs_error:.3f}")
    print(f"RMSE Count Error:           {rmse:.3f}")
    print(f"Mean Bias (det-gt):         {mean_bias:.3f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
