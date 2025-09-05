# 🔍 Real-Time Video Analysis with Kafka and YOLOv5

This project demonstrates a real-time video analysis system using **Apache Kafka** for streaming and **YOLOv5** for object detection. It allows you to capture live video from a mobile device or webcam, stream it via Kafka, and detect objects (like people and helmets) using a YOLOv5 model in real time.

## 🧰 Tech Stack

- **Apache Kafka** – Real-time stream processing

- **YOLOv5** – Object detection model

- **OpenCV** – Frame capture and preprocessing

- **Python** – Backend programming

- **Flask** – Web server (if used for UI/REST)

- **Docker** – (Optional) for containerization

## 📷 Sample Output

| Input Stream | Processed Output |
|--------------|------------------|
| ![Input Frame](images/input_frame.jpg) | ![Detected Output](images/output_frame.jpg) |

> These results show YOLOv5 detecting persons and checking helmet usage in real time.

![Result images1](images/Screenshot%20(127).png) | ![Result image2](images/Screenshot%20(128).png) | ![Result images3](images/Screenshot%20(129).png) | ![Result images4](images/Screenshot%20(131).png) | ![Result images5](images/Screenshot%20(132).png)

## 🚀 How It Works

1. **Producer** captures frames from a camera and sends them to a Kafka topic.
2. **Consumer** listens to that topic, receives frames, and passes them to YOLOv5.
3. The model performs detection and displays results with bounding boxes.
4. Optionally, results can be stored or visualized on a dashboard.

## ✅ Features

- Real-time object detection
- Helmet detection and person counting
- Kafka-based streaming pipeline
- Scalable and modular structure

## ⚙️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/chaitanyya007/HelmetDetection.git
cd HelmetDetection

# Install requirements
pip install -r requirements.txt

# Start Kafka (example for local Kafka setup)
zookeeper-server-start.sh config/zookeeper.properties
kafka-server-start.sh config/server.properties

# Start producer and consumer
python kafka_producer.py
python kafka_consumer.py

---

> **Note:** This repository is a personal copy of the Helmet Detection project  
> jointly developed by **Dikshant Singh** and **Chaitanya Pratap Agarwal**.  
> I cloned it to keep my own record and portfolio of the work we built together.  
> The commit history preserves the original authorship.

