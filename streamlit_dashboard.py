import sys
import pathlib

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

import streamlit as st
from kafka import KafkaConsumer
import base64
import numpy as np
import cv2
import torch
from PIL import Image
import time
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Manually define scale_coords since it's missing from utils
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

# Streamlit setup
st.set_page_config(layout="wide")
st.title("Real-Time YOLOv5 Detection")

status_text = st.empty()
frame_spot = st.empty()

# Paths
yolov5_path = "D:\\Projects+Datsets\\Projects\\Kafka_live_streaming\\yolov5"
weights_path = "D:\\Projects+Datsets\\Projects\\Kafka_live_streaming\\best.pt"


if not os.path.isdir(yolov5_path):
    st.error(f"❌ YOLOv5 directory not found at: {yolov5_path}")
    st.stop()

if not os.path.isfile(weights_path):
    st.error(f"❌ Weights file not found at: {weights_path}")
    st.stop()

if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

try:
    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression
    from utils.torch_utils import select_device

    device = select_device('cpu')
    model = attempt_load(weights_path)  # load without map_location
    model.to(device)                    # send model to device
    model.eval()

    status_text.text("✅ YOLOv5 model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load YOLO model: {e}")
    st.stop()


# Kafka connection
try:
    consumer = KafkaConsumer(
        'video-stream',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    status_text.text("✅ Connected to Kafka topic 'video-stream'")
except Exception as e:
    st.error(f"❌ Failed to connect to Kafka: {e}")
    st.stop()

status_text.text("⏳ Waiting for frames...")

names = model.module.names if hasattr(model, 'module') else model.names

for message in consumer:
    try:
        frame_b64 = message.value
        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            status_text.text("❌ Received empty frame")
            continue

        img = letterbox(frame, 640, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device).float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = model(img_tensor, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        frame_spot.image(img_pil, caption="Live Detection", use_container_width=True)

        status_text.text(f"✅ Frame received and processed at {time.strftime('%H:%M:%S')}")

    except Exception as e:
        status_text.text(f"❌ Error during frame processing: {e}")
