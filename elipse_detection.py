import cv2
import numpy as np
import random
import math
import json
import os
from collections import defaultdict

# --- CONFIGURATION ---
IMAGE_FOLDER = "ellipses"
ANNOTATIONS_FILE = "annotations.json"
WORLD_SPACE_SIZE = 100

# --- RHT PARAMETERS (tuned for more stability) ---
RHT_EPOCHS = 6000
RHT_EPOCHS_HARD = 20000
ACCUMULATOR_BIN_SIZE = 2
MIN_VOTES = 3
TOP_BINS_TO_REFINE = 15

# --- INLIER / SUPPORT PARAMETERS ---
INLIER_THRESHOLD = 2.3
MIN_INLIERS = 16
REFINE_ITERS = 2

# --- MATCHING THRESHOLDS (GT matching for F1) ---
MATCH_THRESH_DIST = 10.0
MATCH_THRESH_AXIS = 0.5

# --- DATASET PRIORS (world units) ---
SEMI_MAJOR_MIN_WORLD = 12.0
SEMI_MAJOR_MAX_WORLD = 22.0
PMMA_X_MIN = 30.0
PMMA_X_MAX = 70.0
PMMA_Y_MIN = 30.0
PMMA_Y_MAX = 70.0

# --- EDGE/POINT SAMPLING ---
MAX_POINTS = 2200
MIN_POINTS_FOR_HARD = 80
CONTOUR_SAMPLE_STEP = 2
MIN_CONTOUR_LEN = 25


def get_scale_factor(img_width: int) -> float:
    return img_width / WORLD_SPACE_SIZE


def convert_to_world(pixel_ellipse, scale: float):
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
    cx = world_ellipse["center_x"]
    cy = world_ellipse["center_y"]
    a = world_ellipse["semi_major_axis"]
    if not (PMMA_X_MIN <= cx <= PMMA_X_MAX):
        return False
    if not (PMMA_Y_MIN <= cy <= PMMA_Y_MAX):
        return False
    if not (SEMI_MAJOR_MIN_WORLD <= a <= SEMI_MAJOR_MAX_WORLD):
        return False
    return True


def preprocess_gray(img, blur_ksize=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if blur_ksize and blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    return gray


def extract_edge_points(img, percentile=60, sigma=0.33):
    gray = preprocess_gray(img, blur_ksize=5)

    g = gray.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    g = (g * 255.0).astype(np.uint8)

    thr = int(np.percentile(g, percentile))
    mask = np.where(g >= thr, g, 0).astype(np.uint8)

    nz = mask[mask > 0]
    v = float(np.median(nz)) if nz.size else float(np.median(mask))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower == upper:
        lower = max(0, lower - 10)
        upper = min(255, upper + 10)

    edges = cv2.Canny(mask, lower, upper)

    # Close small gaps to make contours more meaningful
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contour-balanced sampling of edge points (more stable than all-points random)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in cnts:
        if len(c) < MIN_CONTOUR_LEN:
            continue
        c = c.squeeze(1)
        pts.extend([(int(x), int(y)) for (x, y) in c[::CONTOUR_SAMPLE_STEP]])

    if len(pts) == 0:
        ys, xs = np.where(edges > 0)
        pts = list(zip(xs.tolist(), ys.tolist()))

    # Cap points for speed and to reduce noise domination
    if len(pts) > MAX_POINTS:
        pts = random.sample(pts, MAX_POINTS)

    return pts


def normalize_ellipse(cand):
    (cx, cy), (d1, d2), angle = cand
    if d1 < d2:
        d1, d2 = d2, d1
        angle = (angle + 90) % 180
    return (cx, cy, d1, d2, angle)


def ellipse_sanity(cx, cy, d1, d2, angle, width, height):
    if any(np.isnan(x) for x in [cx, cy, d1, d2, angle]):
        return False
    if not (-width < cx < 2 * width) or not (-height < cy < 2 * height):
        return False
    if d1 < 6 or d2 < 6:
        return False
    if d1 > 4 * max(width, height) or d2 > 4 * max(width, height):
        return False
    return True


def inlier_indices(points, cx, cy, d1, d2, ang_deg):
    ang = ang_deg * math.pi / 180.0
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    a, b = d1 / 2.0, d2 / 2.0
    if a < 0.1 or b < 0.1:
        return []

    tol = INLIER_THRESHOLD / min(a, b)
    idxs = []
    for i, (px, py) in enumerate(points):
        tx, ty = px - cx, py - cy
        xr = tx * cos_a + ty * sin_a
        yr = -tx * sin_a + ty * cos_a
        val = (xr / a) ** 2 + (yr / b) ** 2
        if abs(val - 1.0) < tol:
            idxs.append(i)
    return idxs


def count_inliers(points, cx, cy, d1, d2, ang_deg):
    return len(inlier_indices(points, cx, cy, d1, d2, ang_deg))


def refine_with_inliers(points, ellipse, iters=2):
    if ellipse is None:
        return None, []
    (cx, cy), (d1, d2), ang = ellipse
    for _ in range(iters):
        idxs = inlier_indices(points, cx, cy, d1, d2, ang)
        if len(idxs) < 5:
            return None, []
        inlier_pts = np.array([points[i] for i in idxs], dtype=np.int32)
        try:
            cand = cv2.fitEllipse(inlier_pts)
            cx, cy, d1, d2, ang = normalize_ellipse(cand)
        except Exception:
            return None, []
    final = ((cx, cy), (d1, d2), ang)
    final_idxs = inlier_indices(points, cx, cy, d1, d2, ang)
    return final, final_idxs


def rht_single_pass(points, width, height):
    if len(points) < 5:
        return None, []

    epochs = RHT_EPOCHS if len(points) >= MIN_POINTS_FOR_HARD else RHT_EPOCHS_HARD

    accumulator = defaultdict(int)
    bin_to_params = defaultdict(list)

    for _ in range(epochs):
        sample = random.sample(points, 5)
        try:
            cand = cv2.fitEllipse(np.array(sample, dtype=np.int32))
            cx, cy, d1, d2, angle = normalize_ellipse(cand)

            if not ellipse_sanity(cx, cy, d1, d2, angle, width, height):
                continue

            key = (
                int(cx / ACCUMULATOR_BIN_SIZE),
                int(cy / ACCUMULATOR_BIN_SIZE),
                int(d1 / ACCUMULATOR_BIN_SIZE),
                int(d2 / ACCUMULATOR_BIN_SIZE),
                int(angle / 10),
            )

            accumulator[key] += 1
            bin_to_params[key].append((cx, cy, d1, d2, angle))
        except Exception:
            continue

    if not accumulator:
        return None, []

    sorted_bins = sorted(accumulator.items(), key=lambda kv: kv[1], reverse=True)[:TOP_BINS_TO_REFINE]

    best = None
    best_inliers = -1
    best_votes = -1

    for key, votes in sorted_bins:
        if votes < MIN_VOTES:
            continue

        params = bin_to_params[key]
        avg_cx = float(np.mean([p[0] for p in params]))
        avg_cy = float(np.mean([p[1] for p in params]))
        avg_d1 = float(np.mean([p[2] for p in params]))
        avg_d2 = float(np.mean([p[3] for p in params]))
        avg_ang = float(np.mean([p[4] for p in params]))

        if not ellipse_sanity(avg_cx, avg_cy, avg_d1, avg_d2, avg_ang, width, height):
            continue

        support = count_inliers(points, avg_cx, avg_cy, avg_d1, avg_d2, avg_ang)

        if (support > best_inliers) or (support == best_inliers and votes > best_votes):
            best_inliers = support
            best_votes = votes
            best = (avg_cx, avg_cy, avg_d1, avg_d2, avg_ang)

    if best is None or best_inliers < MIN_INLIERS:
        return None, []

    avg_cx, avg_cy, avg_d1, avg_d2, avg_ang = best
    initial = ((avg_cx, avg_cy), (avg_d1, avg_d2), avg_ang)

    refined, idxs = refine_with_inliers(points, initial, iters=REFINE_ITERS)
    if refined is None or len(idxs) < MIN_INLIERS:
        return None, []

    return refined, idxs


def process_image(img_path, gt_ellipses):
    img = cv2.imread(img_path)
    if img is None:
        return 0, 0, len(gt_ellipses), False

    h, w = img.shape[:2]
    scale = get_scale_factor(w)

    all_points = extract_edge_points(img, percentile=60)

    detected_ellipses = []
    ellipse, _ = rht_single_pass(all_points, w, h)
    if ellipse is not None:
        world_ellipse = convert_to_world(ellipse, scale)
        if ellipse_ok_in_world(world_ellipse):
            detected_ellipses.append(world_ellipse)

    tp = 0
    fp = 0
    matched_gt = set()

    for det in detected_ellipses:
        match_found = False
        for idx, gt in enumerate(gt_ellipses):
            if idx in matched_gt:
                continue

            dist = math.hypot(det["center_x"] - gt["center_x"], det["center_y"] - gt["center_y"])
            if dist > MATCH_THRESH_DIST:
                continue

            if gt["semi_major_axis"] == 0:
                continue

            diff_a = abs(det["semi_major_axis"] - gt["semi_major_axis"]) / gt["semi_major_axis"]
            if diff_a > MATCH_THRESH_AXIS:
                continue

            tp += 1
            matched_gt.add(idx)
            match_found = True
            break

        if not match_found:
            fp += 1

    fn = len(gt_ellipses) - len(matched_gt)

    # Exact-image correctness (for "image accuracy")
    exact_ok = (tp == len(gt_ellipses)) and (fp == 0) and (fn == 0)
    return tp, fp, fn, exact_ok


def main():
    print("--- STARTING PRIMARY RHT (REFINED + BETTER EDGES) ---")
    print(f"Bins: {ACCUMULATOR_BIN_SIZE} | Epochs(base/hard): {RHT_EPOCHS}/{RHT_EPOCHS_HARD} | MinVotes: {MIN_VOTES}")
    print(f"TopBins: {TOP_BINS_TO_REFINE} | MinInliers: {MIN_INLIERS} | RefineIters: {REFINE_ITERS}")

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
    exact_correct = 0

    anns_by_id = {}
    for ann in data["annotations"]:
        anns_by_id[ann["image_id"]] = [e for e in ann["ellipses"] if e.get("type") == "primary"]

    for entry in data["images"]:
        img_id = entry["id"]
        path = os.path.join(IMAGE_FOLDER, f"id_{img_id}.png")
        if not os.path.exists(path):
            path = os.path.join(IMAGE_FOLDER, f"image_{img_id}.png")
        if not os.path.exists(path):
            continue

        gt = anns_by_id.get(img_id, [])
        tp, fp, fn, exact_ok = process_image(path, gt)

        global_tp += tp
        global_fp += fp
        global_fn += fn
        processed += 1
        exact_correct += 1 if exact_ok else 0

        if processed % 50 == 0:
            denom = (global_tp + global_fp + global_fn)
            acc_set = (global_tp / denom) if denom > 0 else 0.0
            acc_img = exact_correct / processed if processed else 0.0
            print(f"Processed {processed} | TP={global_tp} FP={global_fp} FN={global_fn} | Acc(set)={acc_set:.2%} Acc(img)={acc_img:.2%}")

    if processed == 0:
        return

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    denom = (global_tp + global_fp + global_fn)
    accuracy_set = (global_tp / denom) if denom > 0 else 0.0
    accuracy_image = exact_correct / processed if processed else 0.0

    print("\n" + "=" * 34)
    print("FINAL RESULTS (PRIMARY, RHT)")
    print("=" * 34)
    print(f"Images:            {processed}")
    print(f"TP / FP / FN:      {global_tp} / {global_fp} / {global_fn}")
    print(f"Precision:         {precision:.2%}")
    print(f"Recall:            {recall:.2%}")
    print(f"F1 Score:          {f1:.2%}")
    print(f"Accuracy (set):    {accuracy_set:.2%}   (TP/(TP+FP+FN))")
    print(f"Accuracy (image):  {accuracy_image:.2%}   (% images exactly correct)")
    print("=" * 34)


if __name__ == "__main__":
    main()