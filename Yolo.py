import cv2
import torch
import numpy as np
from yolov5 import YOLOv5
from collections import deque

# 加载YOLOv5模型
model = YOLOv5.load("yolov5s.pt")  # 加载YOLOv5模型，可以选择更大版本如yolov5m, yolov5l等

# 图像超分辨率增强（此处简单模拟超分辨率增强）
def enhance_resolution(image):
    # 假设这里已经有一个超分辨率算法，我们直接返回原图进行模拟
    # 实际上你可以使用如ESRGAN、SRCNN等进行超分辨率处理
    return cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

# 目标检测函数，使用YOLOv5进行检测
def detect_objects(image):
    results = model(image)  # 使用YOLOv5进行目标检测
    boxes = results.xyxy[0].cpu().numpy()  # 获取边界框
    labels = results.names  # 获取标签
    confidences = results.confidence[0].cpu().numpy()  # 获取置信度
    return boxes, labels, confidences

# 卡尔曼滤波器（Kalman Tracker）简单实现
class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4维状态，2维观测
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.prev_measurement = None

    def update(self, detection):
        if detection is None:
            return None
        self.kalman.correct(detection)
        prediction = self.kalman.predict()
        return prediction

# 主循环（增强图像 -> 目标检测 -> 跟踪）
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    trackers = []  # 目标跟踪器队列

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 图像增强（超分辨率）
        enhanced_frame = enhance_resolution(frame)

        # 2. 目标检测（使用YOLOv5）
        boxes, labels, confidences = detect_objects(enhanced_frame)

        # 3. 目标跟踪（使用卡尔曼滤波器）
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # 假设我们跟踪目标并绘制检测框
            if confidences[i] > 0.5:  # 只处理置信度大于0.5的检测框
                tracker = KalmanTracker()
                prediction = tracker.update(np.array([x1, y1], dtype=np.float32))
                if prediction is not None:
                    cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(enhanced_frame, f"{labels[i]}: {confidences[i]:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示跟踪结果
        cv2.imshow('Tracking', enhanced_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 调用主函数处理视频流
main('input_video.mp4')
