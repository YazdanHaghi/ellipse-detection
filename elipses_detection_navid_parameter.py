import cv2
import numpy as np
import random
import math
import json
import os
from collections import defaultdict

# =========================================================
# CONFIG (for your 50x50 dataset)
# =========================================================
IMAGE_FOLDER = "ellipses"          # folder with id_0.png, id_1.png, ...
ANNOTATIONS_FILE = "annotations.json"
WORLD_SPACE_SIZE = 100             # world coords (0..100)

# --- RHT PARAMETERS (tuned for small images) ---
RHT_EPOCHS = 3000          # number of random 5-point samples per pass
ACCUMULATOR_BIN_SIZE = 2   # bin size in pixels
MIN_VOTES = 3              # min bin count to accept ellipse
INLIER_THRESHOLD = 2.5     # how tight points must lie on ellipse

# --- MATCHING THRESHOLDS (world units) ---
MATCH_THRESH_DIST = 10.0   # max center distance in world units
MATCH_THRESH_AXIS = 0.5    # 50% rel error on semi-major axis
MATCH_THRESH_ANGLE = 0.6   # (not used yet, kept for future)

# --- SIMPLE PRIORS ON CENTER POSITION (world units) ---
# Taken from your config: PMMA window 30..70 in X and Y
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
    No hard constraint on axes – avoids killing good detections.
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

    1) grayscale
    2) small blur
    3) threshold to remove very dark background
    4) Canny edge detector
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


def angle_diff_deg(rad1, rad2):
    """Smallest absolute difference between two angles in radians, result in degrees."""
    d = abs(rad1 - rad2)
    d = min(d, 2 * math.pi - d)
    return d * 180.0 / math.pi


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

            # quantise parameters into bins
            key = (
                int(cx / ACCUMULATOR_BIN_SIZE),
                int(cy / ACCUMULATOR_BIN_SIZE),
                int(d1 / ACCUMULATOR_BIN_SIZE),
                int(d2 / ACCUMULATOR_BIN_SIZE),
                int(angle / 10),  # 10° bins
            )

            accumulator[key] += 1
            bin_to_params[key].append(((cx, cy), (d1, d2), angle))

        except Exception:
            continue

    if not accumulator:
        return None, []

    best_key = max(accumulator, key=accumulator.get)

    # not enough support
    if accumulator[best_key] < MIN_VOTES:
        return None, []

    cands = bin_to_params[best_key]

    # average parameters in best bin
    avg_cx = np.mean([c[0][0] for c in cands])
    avg_cy = np.mean([c[0][1] for c in cands])
    avg_d1 = np.mean([c[1][0] for c in cands])
    avg_d2 = np.mean([c[1][1] for c in cands])
    avg_ang = np.mean([c[2] for c in cands])

    final_ellipse = ((avg_cx, avg_cy), (avg_d1, avg_d2), avg_ang)

    # ---- inlier selection ----
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
# Per-image processing
# =========================================================
def process_image(img_path, gt_ellipses):
    img = cv2.imread(img_path)
    if img is None:
        return 0, 0, len(gt_ellipses), [], [], [], []

    h, w = img.shape[:2]      # 50x50
    scale = get_scale_factor(w)

    # 1) edge points
    all_points = extract_edge_points(img, intensity_thresh=40)
    detected_ellipses = []
    current_points = all_points[:]

    # 2) RHT (up to 3 ellipses, though in practice you have 1)
    rht_success = False
    for _ in range(3):
        if len(current_points) < 5:
            break

        ellipse, inliers = rht_single_pass(current_points, w, h)
        if ellipse is None:
            break

        world_ellipse = convert_to_world(ellipse, scale)

        # use only center prior, not axis prior
        if not ellipse_ok_in_world(world_ellipse):
            # still remove these inliers – they belong to some structure
            inlier_set = set(inliers)
            current_points = [
                p for i, p in enumerate(current_points) if i not in inlier_set
            ]
            continue

        rht_success = True
        detected_ellipses.append(world_ellipse)

        # remove inliers so we can look for another ellipse
        inlier_set = set(inliers)
        current_points = [
            p for i, p in enumerate(current_points) if i not in inlier_set
        ]

    # 3) fallback: fit a single ellipse to all edge points
    if (not rht_success) and len(all_points) >= 5:
        try:
            pts = np.array([[x, y] for (x, y) in all_points], dtype=np.int32)
            fallback = cv2.fitEllipse(pts)
            world_fb = convert_to_world(fallback, scale)
            if ellipse_ok_in_world(world_fb):
                detected_ellipses.append(world_fb)
        except Exception:
            pass

    # 4) evaluate against GT + collect errors
    tp = 0
    fp = 0
    matched_gt = set()

    center_errors = []
    semi_major_errors = []
    semi_minor_errors = []
    angle_errors = []

    for det in detected_ellipses:
        match_found = False
        for idx, gt in enumerate(gt_ellipses):
            if idx in matched_gt:
                continue

            # center distance in world units
            dist = math.hypot(
                det["center_x"] - gt["center_x"],
                det["center_y"] - gt["center_y"],
            )
            if dist > MATCH_THRESH_DIST:
                continue

            if gt["semi_major_axis"] == 0:
                continue

            # relative error on semi-major axis for matching
            diff_a_rel = abs(det["semi_major_axis"] - gt["semi_major_axis"]) / gt[
                "semi_major_axis"
            ]
            if diff_a_rel > MATCH_THRESH_AXIS:
                continue

            # ---------- this is a true positive ----------
            tp += 1
            matched_gt.add(idx)
            match_found = True

            # collect absolute errors
            center_errors.append(dist)

            semi_major_errors.append(
                abs(det["semi_major_axis"] - gt["semi_major_axis"])
            )
            semi_minor_errors.append(
                abs(det["semi_minor_axis"] - gt["semi_minor_axis"])
            )

            angle_errors.append(
                angle_diff_deg(
                    det["orientation_angle_rad"], gt["orientation_angle_rad"]
                )
            )

            break

        if not match_found:
            fp += 1

    fn = len(gt_ellipses) - len(matched_gt)
    return tp, fp, fn, center_errors, semi_major_errors, semi_minor_errors, angle_errors


# =========================================================
# Main evaluation loop
# =========================================================
def main():
    print(f"--- STARTING RHT ON 50x50 DATASET ---")
    print(
        f"Bins: {ACCUMULATOR_BIN_SIZE} | Epochs: {RHT_EPOCHS} | Min Votes: {MIN_VOTES}"
    )

    try:
        with open(ANNOTATIONS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        print("JSON not found.")
        return

    global_tp = 0
    global_fp = 0
    global_fn = 0
    processed = 0

    all_center_errors = []
    all_semi_major_errors = []
    all_semi_minor_errors = []
    all_angle_errors = []

    # group annotations by image id, keep only 'primary'
    anns_by_id = {}
    for ann in data["annotations"]:
        anns_by_id[ann["image_id"]] = [
            e for e in ann["ellipses"] if e.get("type") == "primary"
        ]

    for entry in data["images"]:
        img_id = entry["id"]
        path = os.path.join(IMAGE_FOLDER, f"id_{img_id}.png")
        if not os.path.exists(path):
            path = os.path.join(IMAGE_FOLDER, f"image_{img_id}.png")
        if not os.path.exists(path):
            continue

        gt = anns_by_id.get(img_id, [])
        (
            tp,
            fp,
            fn,
            center_errs,
            a_errs,
            b_errs,
            ang_errs,
        ) = process_image(path, gt)

        global_tp += tp
        global_fp += fp
        global_fn += fn
        processed += 1

        all_center_errors.extend(center_errs)
        all_semi_major_errors.extend(a_errs)
        all_semi_minor_errors.extend(b_errs)
        all_angle_errors.extend(ang_errs)

        if processed % 50 == 0:
            print(
                f"Processed {processed} images... (TP: {global_tp}, FN: {global_fn})"
            )

    if processed == 0:
        return

    precision = (
        global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    )
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # mean errors over all matched ellipses
    def mean_or_zero(arr):
        return sum(arr) / len(arr) if arr else 0.0

    mean_center_error = mean_or_zero(all_center_errors)
    mean_a_error = mean_or_zero(all_semi_major_errors)
    mean_b_error = mean_or_zero(all_semi_minor_errors)
    mean_angle_error = mean_or_zero(all_angle_errors)

    print("\n" + "=" * 30)
    print("FINAL RESULTS")
    print("=" * 30)
    print(f"Images:    {processed}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("-" * 30)
    print(f"Mean Error Center:      {mean_center_error:.3f} (world units)")
    print(f"Mean Error Semi-major:  {mean_a_error:.3f} (world units)")
    print(f"Mean Error Semi-minor:  {mean_b_error:.3f} (world units)")
    print(f"Mean Error Angle:       {mean_angle_error:.3f} (degrees)")
    print("=" * 30)


if __name__ == "__main__":
    main()
