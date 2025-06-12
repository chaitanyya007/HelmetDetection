from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType, StringType
import numpy as np
import cv2
import torch

# Initialize Spark
spark = SparkSession.builder \
    .appName("YOLO Kafka Stream") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load YOLO model (do this inside a UDF if running on worker nodes)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define UDF to run detection
def detect_objects(frame_bytes):
    img_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return "Frame error"

    results = model(frame)
    labels = results.names
    detections = results.xyxy[0]
    detected_classes = [labels[int(d[5])] for d in detections]

    return ', '.join(detected_classes)

detect_udf = udf(detect_objects, StringType())

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "video-stream") \
    .load()

# Apply detection
decoded_df = df.withColumn("detected_objects", detect_udf(col("value")))

# Write results to console (or to DB, file, etc.)
query = decoded_df.select("detected_objects") \
    .writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

query.awaitTermination()
