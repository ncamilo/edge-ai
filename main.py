import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from collections import defaultdict
from PIL import Image
import io
from ultralytics import YOLO
from io import BytesIO
import numpy as np

import cv2
from fastapi.responses import Response


app = FastAPI(debug=True)

model = YOLO("yolov8n.pt")

# Defina um modelo Pydantic para o corpo da requisição JSON
class DetectRequest(BaseModel):
    image: str  # URL da imagem
    conf: float = 0.8


@app.post("/detect/json")
async def detect_json(req: DetectRequest):

    # Download da imagem
    try:
        resp = requests.get(req.image, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "failed_to_fetch_image",
                "message": str(e),
                "image_url": req.image
            }
        )

    
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    img_np = np.array(img)

    # Inferência
    start = time.perf_counter()
    results = model(img_np, conf=req.conf)
    inference_time_ms = (time.perf_counter() - start) * 1000

    detections = []
    count_by_class = defaultdict(int)
    classes_detected = set()

    # Processar resultados
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(float, box.xyxy[0])

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

            classes_detected.add(cls_name)
            count_by_class[cls_name] += 1

    # Retorno JSON
    return {
        "image_url": req.image,
        "model": model.model_name if hasattr(model, "model_name") else "yolov8",
        "inference_time_ms": round(inference_time_ms * 1000, 2),
        "detections": detections,
        "classes_detected": sorted(list(classes_detected)),
        "count_by_class": dict(count_by_class)
    }


@app.post("/detect/image")
def detect_image(req: DetectRequest):

    # Download da imagem
    try:
        resp = requests.get(req.image, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "failed_to_fetch_image",
                "message": str(e),
                "image_url": req.image
            }
        )

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    img_np = np.array(img)

    # Converter RGB -> BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Inferência
    start = time.perf_counter()
    results = model(img_np, conf=req.conf)
    inference_time_ms = (time.perf_counter() - start) * 1000

     # Desenhar bounding boxes
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{cls_name} {confidence:.2f}"

            # Bounding box
            cv2.rectangle(
                img_bgr,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label background
            (w, h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                img_bgr,
                (x1, y1 - h - 10),
                (x1 + w, y1),
                (0, 255, 0),
                -1
            )

            # Label text
            cv2.putText(
                img_bgr,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

    # Converter para PNG
    success, png = cv2.imencode(".png", img_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Erro ao gerar imagem PNG")

    # Retornar imagem
    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={
            "X-Inference-Time-ms": f"{inference_time_ms:.2f}"
        }
    )    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
