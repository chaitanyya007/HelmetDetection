from kafka import KafkaConsumer
import cv2
import base64
import numpy as np
import torch

# Kafka Consumer setup
consumer = KafkaConsumer(
    'video-stream',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True
)

# Load YOLO model (use YOLOv5 for simplicity, change to your model if needed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5s is a smaller model

print("[INFO] Waiting for frames from Kafka...")

for message in consumer:
    try:
        # Decode the base64-encoded JPEG frame
        jpg_original = base64.b64decode(message.value)
        jpg_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)

        # Perform YOLO inference on the frame
        results = model(frame)
        frame_with_detections = results.render()[0]  # Rendered frame with detections

        # Display the frame with detections
        cv2.imshow("Consumer View", frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"[ERROR] Failed to decode/display frame: {e}")

cv2.destroyAllWindows()
