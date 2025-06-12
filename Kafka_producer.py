from kafka import KafkaProducer
import cv2
import base64
import time

# Set this to your phone's IP
VIDEO_STREAM_URL = 'http://172.16.6.240:8080/video'

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: base64.b64encode(x)
)

cap = cv2.VideoCapture(VIDEO_STREAM_URL)

if not cap.isOpened():
    print("❌ Failed to open video stream. Check IP/Webcam app.")
    exit()

print("✅ Connected to IP Webcam stream.")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("⚠️ Frame not read properly, retrying...")
        time.sleep(0.1)
        continue

    _, buffer = cv2.imencode('.jpg', frame)
    producer.send('video-stream', buffer.tobytes())
    frame_count += 1

    if frame_count % 10 == 0:
        print(f"✅ Sent {frame_count} frames")

    time.sleep(0.1)  # ~10 FPS
