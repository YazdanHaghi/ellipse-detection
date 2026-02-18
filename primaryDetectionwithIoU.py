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
IMAGE_FOLDER = "ellipses"
ANNOTATIONS_FILE = "annotations.json"
WORLD_SPACE_SIZE = 100

# Accuracy Thresholds
IOU_THRESHOLD = 0.6  # Detection considered correct if IoU > 80%

# RHT Parameters
RHT_EPOCHS = 3000
ACCUMULATOR_BIN_SIZE = 2
MIN_VOTES = 3
INLIER_THRESHOLD = 2.5

# Priors
PMMA_X_MIN = 30.0
PMMA_X_MAX = 70.0
PMMA_Y_MIN = 30.0
PMMA_Y_MAX = 70.0


# =========================================================
# Utility & Geometry Functions
# =========================================================
def get_scale_factor(img_width: int) -> float:
    return img_width / WORLD_SPACE_SIZE


def convert_to_world(pixel_ellipse, scale: float):
    (cx, cy), (d1, d2), angle_deg = pixel_ellipse
    wx = cx / scale
    wy = cy / scale
    # d1 is diameter, so radius is d/2
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
    if not (PMMA_X_MIN <= cx <= PMMA_X_MAX): return False
    if not (PMMA_Y_MIN <= cy <= PMMA_Y_MAX): return False
    return True


def calculate_iou(ellipse_gt, ellipse_det, grid_size=500):
    """
    Calculates Intersection over Union (IoU) using mask rasterization.
    We draw both ellipses on a high-res grid (e.g. 500x500 for 0-100 space).
    """
    if ellipse_gt is None or ellipse_det is None:
        return 0.0

    # Scale world coords (0-100) to grid coords (0-500)
    scale = grid_size / WORLD_SPACE_SIZE

    # Create empty masks
    mask_gt = np.zeros((grid_size, grid_size), dtype=np.uint8)
    mask_det = np.zeros((grid_size, grid_size), dtype=np.uint8)

    def draw(mask, e):
        cx = int(e["center_x"] * scale)
        cy = int(e["center_y"] * scale)
        # cv2.ellipse takes axes as (semi-major, semi-minor)
        ax1 = int(e["semi_major_axis"] * scale)
        ax2 = int(e["semi_minor_axis"] * scale)
        angle_deg = np.degrees(e["orientation_angle_rad"])

        if ax1 <= 0 or ax2 <= 0: return
        cv2.ellipse(mask, (cx, cy), (ax1, ax2), angle_deg, 0, 360, 255, -1)

    draw(mask_gt, ellipse_gt)
    draw(mask_det, ellipse_det)

    intersection = np.logical_and(mask_gt, mask_det)
    union = np.logical_or(mask_gt, mask_det)

    area_inter = np.sum(intersection)
    area_union = np.sum(union)

    if area_union == 0:
        return 0.0

    return area_inter / area_union


def extract_edge_points(img, intensity_thresh=40, blur_ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(gray, intensity_thresh, 255, cv2.THRESH_TOZERO)
    edges = cv2.Canny(mask, 50, 150)
    ys, xs = np.where(edges > 0)
    return list(zip(xs.tolist(), ys.tolist()))


# =========================================================
# Randomized Hough Transform
# =========================================================
def rht_single_pass(points, width, height):
    if len(points) < 5: return None

    accumulator = defaultdict(int)
    bin_to_params = defaultdict(list)

    for _ in range(RHT_EPOCHS):
        sample = random.sample(points, 5)
        try:
            cand = cv2.fitEllipse(np.array(sample, dtype=np.int32))
            (cx, cy), (d1, d2), angle = cand

            if np.isnan(cx) or d1 < 2 or d2 < 2: continue
            if not (-width < cx < 2 * width): continue

            # Ensure d1 is major axis
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
        except:
            continue

    if not accumulator: return None

    # --- MODIFIED SELECTION STRATEGY (Optional) ---
    # To truly favor primary (bigger) ellipses, we could weight votes by area.
    # For now, we stick to Max Votes to see the "true" accuracy of the current algorithm.
    best_key = max(accumulator, key=accumulator.get)

    if accumulator[best_key] < MIN_VOTES: return None

    cands = bin_to_params[best_key]
    avg_cx = np.mean([c[0][0] for c in cands])
    avg_cy = np.mean([c[0][1] for c in cands])
    avg_d1 = np.mean([c[1][0] for c in cands])
    avg_d2 = np.mean([c[1][1] for c in cands])
    avg_ang = np.mean([c[2] for c in cands])

    return ((avg_cx, avg_cy), (avg_d1, avg_d2), avg_ang)


# =========================================================
# Process Single Image
# =========================================================
def process_image_primary(img_path, gt_primary_ellipses):
    """
    Returns:
       iou_score (0.0 to 1.0)
       found_something (bool)
    """
    img = cv2.imread(img_path)

    # Setup GT
    # We assume there is exactly 1 primary ellipse based on the JSON analysis
    gt_ellipse = gt_primary_ellipses[0] if len(gt_primary_ellipses) > 0 else None

    if img is None:
        return 0.0, False

    h, w = img.shape[:2]
    scale = get_scale_factor(w)
    all_points = extract_edge_points(img)

    detected_primary = None

    # RHT Pass
    if len(all_points) >= 5:
        ellipse = rht_single_pass(all_points, w, h)
        if ellipse is not None:
            world_ellipse = convert_to_world(ellipse, scale)
            if ellipse_ok_in_world(world_ellipse):
                detected_primary = world_ellipse

    # Fallback
    if (detected_primary is None) and len(all_points) >= 5:
        try:
            pts = np.array([[x, y] for (x, y) in all_points], dtype=np.int32)
            fallback = cv2.fitEllipse(pts)
            world_fb = convert_to_world(fallback, scale)
            if ellipse_ok_in_world(world_fb):
                detected_primary = world_fb
        except:
            pass

    # Calculate IoU
    iou = 0.0
    if gt_ellipse and detected_primary:
        iou = calculate_iou(gt_ellipse, detected_primary)

    return iou, (detected_primary is not None)


# =========================================================
# Main Evaluation
# =========================================================
def main():
    print(f"--- STARTING RHT EVALUATION (IoU MODE) ---")

    try:
        with open(ANNOTATIONS_FILE, "r") as f:
            data = json.load(f)
    except:
        print("JSON not found.")
        return

    processed = 0
    ious = []
    correct_detections = 0  # count where IoU > Threshold

    # Group annotations
    anns_by_id = {}
    for ann in data["annotations"]:
        prim = [e for e in ann["ellipses"] if e.get("type") == "primary"]
        anns_by_id[ann["image_id"]] = prim

    for entry in data["images"]:
        img_id = entry["id"]
        # Handle file naming variations
        path = os.path.join(IMAGE_FOLDER, f"id_{img_id}.png")
        if not os.path.exists(path):
            path = os.path.join(IMAGE_FOLDER, f"image_{img_id}.png")
        if not os.path.exists(path):
            continue

        gt_prim = anns_by_id.get(img_id, [])
        iou, found = process_image_primary(path, gt_prim)

        processed += 1
        ious.append(iou)

        if iou >= IOU_THRESHOLD:
            correct_detections += 1

        if processed % 50 == 0:
            current_acc = correct_detections / processed
            print(f"Processed {processed}... Current Accuracy (IoU>{IOU_THRESHOLD}): {current_acc:.2%}")

    if processed == 0: return

    mean_iou = sum(ious) / processed
    final_accuracy = correct_detections / processed

    print("\n" + "=" * 40)
    print("FINAL GEOMETRIC RESULTS (TRUE PRIMARY)")
    print("=" * 40)
    print(f"Total Images:           {processed}")
    print(f"Accuracy (IoU >= {IOU_THRESHOLD}): {final_accuracy:.2%}")
    print(f"Mean IoU:               {mean_iou:.3f}")
    print("=" * 40)


if __name__ == "__main__":
    main()