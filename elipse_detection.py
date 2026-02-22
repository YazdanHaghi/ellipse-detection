import cv2
import numpy as np
import random
import math
import json
import os
from collections import defaultdict

# -------------------------
# CONFIG
# -------------------------
IMAGE_FOLDER = "ellipses"
ANNOTATIONS_FILE = "annotations.json"
WORLD_SPACE_SIZE = 100

# Evaluation (keep your strict setting if you want)
MATCH_THRESH_DIST = 5.0
MATCH_THRESH_AXIS = 0.5

# Priors (world)
SEMI_MAJOR_MIN_WORLD = 12.0
SEMI_MAJOR_MAX_WORLD = 22.0
PMMA_X_MIN = 30.0
PMMA_X_MAX = 70.0
PMMA_Y_MIN = 30.0
PMMA_Y_MAX = 70.0

# Candidate generation
MAX_CONTOURS = 12
TOP_RHT_BINS = 25
TOP_RHT_CANDS = 10

RHT_EPOCHS = 6000
RHT_EPOCHS_HARD = 18000
ACCUMULATOR_BIN_SIZE = 2
MIN_VOTES = 2

# Inlier/support
INLIER_THRESHOLD = 2.5
MIN_INLIERS_BASE = 8

# Refinement
REFINE_ITERS = 2

# Training (auto-tune weights)
DO_AUTOTUNE = True
TRAIN_SPLIT = 0.8
RANDOM_SEED = 7
WEIGHT_TRIALS = 2500  # increase for better tuning if runtime allows


# -------------------------
# Utils
# -------------------------
def get_scale_factor(img_width: int) -> float:
    return img_width / WORLD_SPACE_SIZE


def normalize_ellipse(cand):
    (cx, cy), (d1, d2), angle = cand
    if d1 < d2:
        d1, d2 = d2, d1
        angle = (angle + 90) % 180
    return (float(cx), float(cy), float(d1), float(d2), float(angle))


def convert_to_world_params(cx, cy, d1, d2, angle_deg, scale):
    wx = cx / scale
    wy = cy / scale
    a = (d1 / 2.0) / scale
    b = (d2 / 2.0) / scale
    return wx, wy, max(a, b), min(a, b), angle_deg * (math.pi / 180.0)


def match_det_to_gt(det_world, gt_list):
    dx, dy, da = det_world  # center_x, center_y, semi_major_axis
    for gt in gt_list:
        dist = math.hypot(dx - gt["center_x"], dy - gt["center_y"])
        if dist > MATCH_THRESH_DIST:
            continue
        g_a = gt.get("semi_major_axis", 0.0)
        if g_a <= 0:
            continue
        rel = abs(da - g_a) / g_a
        if rel > MATCH_THRESH_AXIS:
            continue
        return True
    return False


def count_inliers(points, cx, cy, d1, d2, ang_deg):
    ang = ang_deg * math.pi / 180.0
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    a, b = d1 / 2.0, d2 / 2.0
    if a < 0.1 or b < 0.1:
        return 0
    tol = INLIER_THRESHOLD / min(a, b)

    c = 0
    for (px, py) in points:
        tx, ty = px - cx, py - cy
        xr = tx * cos_a + ty * sin_a
        yr = -tx * sin_a + ty * cos_a
        val = (xr / a) ** 2 + (yr / b) ** 2
        if abs(val - 1.0) < tol:
            c += 1
    return c


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


def refine_with_inliers(points, params):
    cx, cy, d1, d2, ang = params
    for _ in range(REFINE_ITERS):
        idxs = inlier_indices(points, cx, cy, d1, d2, ang)
        if len(idxs) < 25:
            break
        pts = np.array([points[i] for i in idxs], dtype=np.int32).reshape(-1, 1, 2)
        try:
            if hasattr(cv2, "fitEllipseAMS"):
                e2 = cv2.fitEllipseAMS(pts)
            else:
                e2 = cv2.fitEllipse(pts)
            cx2, cy2, d1_2, d2_2, ang2 = normalize_ellipse(e2)

            sup_old = count_inliers(points, cx, cy, d1, d2, ang)
            sup_new = count_inliers(points, cx2, cy2, d1_2, d2_2, ang2)

            if sup_new >= sup_old:
                cx, cy, d1, d2, ang = cx2, cy2, d1_2, d2_2, ang2
            else:
                break
        except Exception:
            break
    return (cx, cy, d1, d2, ang)


# -------------------------
# Preprocess variants
# -------------------------
def preprocess_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    out = []

    # Otsu normal
    _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th1)

    # Otsu inverted
    out.append(255 - th1)

    # Adaptive mean
    th2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 2)
    out.append(th2)
    out.append(255 - th2)

    # Adaptive gaussian
    th3 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)
    out.append(th3)
    out.append(255 - th3)

    cleaned = []
    kernel = np.ones((3, 3), np.uint8)
    for th in out:
        t = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned.append(t)

    return cleaned


def edge_points_from_mask(mask):
    edges = cv2.Canny(mask, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    ys, xs = np.where(edges > 0)
    return list(zip(xs.tolist(), ys.tolist()))


# -------------------------
# Candidate generation
# -------------------------
def contour_candidates(img):
    cands = []
    for mask in preprocess_variants(img):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS]
        for c in cnts:
            if len(c) < 30:
                continue
            try:
                if hasattr(cv2, "fitEllipseAMS"):
                    e = cv2.fitEllipseAMS(c)
                else:
                    e = cv2.fitEllipse(c)
                cx, cy, d1, d2, ang = normalize_ellipse(e)
                area = float(cv2.contourArea(c))
                cands.append((cx, cy, d1, d2, ang, area, "contour"))
            except Exception:
                continue
    return cands


def rht_candidates(points, width, height):
    if len(points) < 30:
        return []

    epochs = RHT_EPOCHS if len(points) >= 180 else RHT_EPOCHS_HARD
    accumulator = defaultdict(int)
    bin_to_params = defaultdict(list)

    for _ in range(epochs):
        sample = random.sample(points, 5)
        try:
            e = cv2.fitEllipse(np.array(sample, dtype=np.int32))
            cx, cy, d1, d2, ang = normalize_ellipse(e)

            key = (
                int(cx / ACCUMULATOR_BIN_SIZE),
                int(cy / ACCUMULATOR_BIN_SIZE),
                int(d1 / ACCUMULATOR_BIN_SIZE),
                int(d2 / ACCUMULATOR_BIN_SIZE),
                int(ang / 10),
            )
            accumulator[key] += 1
            bin_to_params[key].append((cx, cy, d1, d2, ang))
        except Exception:
            continue

    if not accumulator:
        return []

    sorted_bins = sorted(accumulator.items(), key=lambda kv: kv[1], reverse=True)[:TOP_RHT_BINS]

    cands = []
    for key, votes in sorted_bins:
        if votes < MIN_VOTES:
            continue
        params = bin_to_params[key]
        cx = float(np.mean([p[0] for p in params]))
        cy = float(np.mean([p[1] for p in params]))
        d1 = float(np.mean([p[2] for p in params]))
        d2 = float(np.mean([p[3] for p in params]))
        ang = float(np.mean([p[4] for p in params]))
        cands.append((cx, cy, d1, d2, ang, float(votes), "rht"))

    cands.sort(key=lambda x: x[5], reverse=True)
    return cands[:TOP_RHT_CANDS]


# -------------------------
# Features + scoring
# -------------------------
def compute_features(cx, cy, d1, d2, ang, aux, method, points, scale):
    a_px = d1 / 2.0
    b_px = d2 / 2.0
    ratio = a_px / (b_px + 1e-6)
    if ratio < 1.0:
        ratio = 1.0 / ratio

    ratio_pen = 0.0
    if ratio < 1.05:
        ratio_pen = (1.05 - ratio)
    elif ratio > 2.8:
        ratio_pen = (ratio - 2.8)

    exp_cx = 50.0 * scale
    exp_cy = 50.0 * scale
    center_dist = math.hypot(cx - exp_cx, cy - exp_cy)

    wx, wy, wa, wb, _ = convert_to_world_params(cx, cy, d1, d2, ang, scale)

    prior_pen = 0.0
    if wx < PMMA_X_MIN:
        prior_pen += (PMMA_X_MIN - wx)
    elif wx > PMMA_X_MAX:
        prior_pen += (wx - PMMA_X_MAX)
    if wy < PMMA_Y_MIN:
        prior_pen += (PMMA_Y_MIN - wy)
    elif wy > PMMA_Y_MAX:
        prior_pen += (wy - PMMA_Y_MAX)
    if wa < SEMI_MAJOR_MIN_WORLD:
        prior_pen += (SEMI_MAJOR_MIN_WORLD - wa) * 2.0
    elif wa > SEMI_MAJOR_MAX_WORLD:
        prior_pen += (wa - SEMI_MAJOR_MAX_WORLD) * 2.0

    support = count_inliers(points, cx, cy, d1, d2, ang)

    # aux: contour area or votes
    if method == "contour":
        area = float(aux)
        votes = 0.0
    else:
        area = 0.0
        votes = float(aux)

    # Feature vector (all positive)
    # We maximize: +support +a_px +area +votes  minus penalties
    feats = np.array([
        float(support),
        float(a_px),
        float(area),
        float(votes),
        float(center_dist),
        float(ratio_pen),
        float(prior_pen),
        1.0
    ], dtype=np.float32)

    return feats, support, (wx, wy, wa)


def pick_primary(candidates, points, scale, weights):
    best = None
    best_score = -1e18

    min_inliers = max(MIN_INLIERS_BASE, int(0.03 * len(points))) if len(points) > 0 else MIN_INLIERS_BASE

    for (cx, cy, d1, d2, ang, aux, method) in candidates:
        feats, support, _ = compute_features(cx, cy, d1, d2, ang, aux, method, points, scale)
        if support < min_inliers:
            continue
        score = float(np.dot(weights, feats))
        if score > best_score:
            best_score = score
            best = (cx, cy, d1, d2, ang, method)

    if best is None:
        return None

    cx, cy, d1, d2, ang, method = best
    refined = refine_with_inliers(points, (cx, cy, d1, d2, ang))
    cx, cy, d1, d2, ang = refined
    return ((cx, cy), (d1, d2), ang)


# -------------------------
# Pipeline per image
# -------------------------
def detect_primary(img):
    h, w = img.shape[:2]
    scale = get_scale_factor(w)

    # Collect candidates (do NOT hard-reject here)
    cands = []
    cands.extend(contour_candidates(img))

    # Edge points for scoring + RHT fallback
    # Use union of edges from multiple masks to increase coverage
    points = []
    for mask in preprocess_variants(img):
        points.extend(edge_points_from_mask(mask))
    if len(points) > 6000:
        points = random.sample(points, 6000)

    # Add RHT candidates
    if len(points) >= 30:
        cands.extend(rht_candidates(points, w, h))

    return cands, points, scale


def evaluate_dataset(image_ids, paths_by_id, gt_by_id, weights):
    tp = fp = fn = 0
    correct_images = 0
    processed = 0

    for img_id in image_ids:
        path = paths_by_id.get(img_id)
        if path is None or not os.path.exists(path):
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        processed += 1
        gt = gt_by_id.get(img_id, [])

        candidates, points, scale = detect_primary(img)
        det = pick_primary(candidates, points, scale, weights)

        det_list = []
        if det is not None:
            cx, cy = det[0]
            d1, d2 = det[1]
            ang = det[2]
            wx, wy, wa, _, _ = convert_to_world_params(cx, cy, d1, d2, ang, scale)
            det_list.append((wx, wy, wa))

        matched = False
        if len(det_list) > 0 and len(gt) > 0:
            matched = match_det_to_gt(det_list[0], gt)

        if matched:
            tp += 1
            correct_images += 1
        else:
            # If we produced a detection but it didn't match => FP
            if len(det_list) > 0:
                fp += 1
            # If GT exists and we missed => FN
            if len(gt) > 0:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = correct_images / processed if processed > 0 else 0.0

    return {
        "processed": processed,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def autotune_weights(train_ids, paths_by_id, gt_by_id):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # weights correspond to:
    # [support, a_px, area, votes, center_dist, ratio_pen, prior_pen, bias]
    # score = +support +a +area +votes -center -ratio_pen -prior_pen +bias
    # We'll sample weights with correct sign tendencies but allow variation.
    best_w = None
    best_acc = -1.0

    def sample_w():
        w = np.zeros(8, dtype=np.float32)
        w[0] = np.random.uniform(0.7, 1.6)     # support +
        w[1] = np.random.uniform(0.0, 0.20)    # size +
        w[2] = np.random.uniform(0.0, 0.0008)  # contour area + (small scale)
        w[3] = np.random.uniform(0.0, 1.0)     # votes +
        w[4] = -np.random.uniform(0.005, 0.08) # center -
        w[5] = -np.random.uniform(2.0, 25.0)   # ratio penalty -
        w[6] = -np.random.uniform(0.2, 6.0)    # prior penalty -
        w[7] = np.random.uniform(-5.0, 5.0)    # bias
        return w

    # Start with a sane default
    default_w = np.array([1.2, 0.08, 0.00025, 0.4, -0.03, -10.0, -1.5, 0.0], dtype=np.float32)
    best_w = default_w
    best_acc = evaluate_dataset(train_ids, paths_by_id, gt_by_id, best_w)["accuracy"]

    for t in range(WEIGHT_TRIALS):
        w = sample_w()
        res = evaluate_dataset(train_ids, paths_by_id, gt_by_id, w)
        acc = res["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_w = w

    return best_w, best_acc


def main():
    with open(ANNOTATIONS_FILE, "r") as f:
        data = json.load(f)

    gt_by_id = {}
    for ann in data["annotations"]:
        gt_by_id[ann["image_id"]] = [e for e in ann["ellipses"] if e.get("type") == "primary"]

    paths_by_id = {}
    image_ids = []
    for entry in data["images"]:
        img_id = entry["id"]
        p1 = os.path.join(IMAGE_FOLDER, f"id_{img_id}.png")
        p2 = os.path.join(IMAGE_FOLDER, f"image_{img_id}.png")
        path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
        if path is None:
            continue
        paths_by_id[img_id] = path
        image_ids.append(img_id)

    if not image_ids:
        print("No images found.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(image_ids)
    split = int(TRAIN_SPLIT * len(image_ids))
    train_ids = image_ids[:split]
    val_ids = image_ids[split:]

    if DO_AUTOTUNE:
        best_w, train_acc = autotune_weights(train_ids, paths_by_id, gt_by_id)
    else:
        best_w = np.array([1.2, 0.08, 0.00025, 0.4, -0.03, -10.0, -1.5, 0.0], dtype=np.float32)
        train_acc = evaluate_dataset(train_ids, paths_by_id, gt_by_id, best_w)["accuracy"]

    train_res = evaluate_dataset(train_ids, paths_by_id, gt_by_id, best_w)
    val_res = evaluate_dataset(val_ids, paths_by_id, gt_by_id, best_w)

    print("\n" + "=" * 34)
    print("TRAIN RESULTS")
    print("=" * 34)
    print(f"Images:    {train_res['processed']}")
    print(f"Accuracy:  {train_res['accuracy']:.2%}")
    print(f"Precision: {train_res['precision']:.2%}")
    print(f"Recall:    {train_res['recall']:.2%}")
    print(f"F1 Score:  {train_res['f1']:.2%}")
    print(f"Weights:   {best_w.tolist()}")

    print("\n" + "=" * 34)
    print("VALIDATION RESULTS")
    print("=" * 34)
    print(f"Images:    {val_res['processed']}")
    print(f"Accuracy:  {val_res['accuracy']:.2%}")
    print(f"Precision: {val_res['precision']:.2%}")
    print(f"Recall:    {val_res['recall']:.2%}")
    print(f"F1 Score:  {val_res['f1']:.2%}")
    print("=" * 34)


if __name__ == "__main__":
    main()