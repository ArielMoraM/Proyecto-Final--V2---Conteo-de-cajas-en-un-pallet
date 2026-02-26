import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

TEXT_PROMPT = "a cardboard box. a carton box. stacked boxes. shipping box. warehouse cardboard boxes"
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.30
MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conteo de cajas usando Grounding DINO + SAM")
    parser.add_argument("--video-path", required=True, help="Ruta del video de entrada")
    parser.add_argument("--frames-dir", default="./frames", help="Carpeta para guardar frames")
    parser.add_argument("--max-frames", type=int, default=8, help="Cantidad máxima de frames a procesar")
    parser.add_argument("--sam-checkpoint", default="./checkpoints/sam_vit_b.pth", help="Ruta del checkpoint SAM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda o cpu")
    return parser.parse_args()


def ensure_sam_checkpoint(checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return

    import urllib.request

    print(f"Descargando checkpoint SAM en {checkpoint_path}...")
    urllib.request.urlretrieve(SAM_CHECKPOINT_URL, checkpoint_path)


def extract_frames(video_path: str, frames_dir: str, max_frames: int) -> List[str]:
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    frame_paths = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(frame, (640, 360))
        save_path = os.path.join(frames_dir, f"frame_{count:03d}.jpg")
        cv2.imwrite(save_path, frame_small)
        frame_paths.append(save_path)
        count += 1

    cap.release()
    print(f"Frames guardados: {count}")
    return frame_paths


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    if len(boxes) == 0:
        return []

    order = np.argsort(-scores)
    keep: List[int] = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        remaining = []
        for j in order[1:]:
            if box_iou(boxes[i], boxes[j]) < iou_threshold:
                remaining.append(j)
        order = np.array(remaining, dtype=int)

    return keep


class PalletCounter:
    def __init__(self, checkpoint_path: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def dino_detect_boxes(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, text=TEXT_PROMPT, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=target_sizes,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )[0]

        boxes = results["boxes"].detach().cpu().numpy() if len(results["boxes"]) else np.empty((0, 4))
        scores = results["scores"].detach().cpu().numpy() if len(results["scores"]) else np.array([])
        labels = [str(l) for l in results["labels"]]

        if len(boxes) > 0:
            keep = nms(boxes, scores, iou_threshold=0.45)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = [labels[i] for i in keep]

        return boxes, scores, labels

    def sam_dino_boxes_for_frame(self, image_rgb: np.ndarray):
        # Esta función faltaba en el notebook original y causaba el error de ejecución.
        dino_boxes, dino_scores, dino_labels = self.dino_detect_boxes(image_rgb)
        semantic_masks = []

        if len(dino_boxes) == 0:
            return semantic_masks, dino_boxes, dino_scores, dino_labels

        self.predictor.set_image(image_rgb)

        for box in dino_boxes:
            masks, mask_scores, _ = self.predictor.predict(
                box=box,
                multimask_output=True,
            )
            best_mask = masks[np.argmax(mask_scores)]
            semantic_masks.append(best_mask.astype(np.uint8))

        return semantic_masks, dino_boxes, dino_scores, dino_labels


def main() -> None:
    args = parse_args()
    ensure_sam_checkpoint(Path(args.sam_checkpoint))

    frame_paths = extract_frames(args.video_path, args.frames_dir, args.max_frames)
    if not frame_paths:
        raise RuntimeError("No se extrajeron frames. Verifica la ruta del video.")

    counter = PalletCounter(checkpoint_path=args.sam_checkpoint, device=args.device)

    conteos_por_frame = []
    for fp in frame_paths:
        img_bgr = cv2.imread(fp)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        semantic_masks, _, _, _ = counter.sam_dino_boxes_for_frame(img_rgb)
        conteos_por_frame.append(len(semantic_masks))
        print(f"{Path(fp).name} -> {len(semantic_masks)}")

    print("Promedio:", round(float(np.mean(conteos_por_frame)), 2))


if __name__ == "__main__":
    main()
