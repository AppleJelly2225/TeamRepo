import cv2
import torch
import numpy as np
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

DISPLAY_USE_HEADLESS = True
CONF_THRESHOLD = 0.3
ONLY_WHITE_THRESHOLD = 180

class TargetTracker:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Currently using {device}")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.model.conf = CONF_THRESHOLD

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open camera")

        self.target_box = None
        self.target_locked = False

    def is_wearing_white(self, crop):
        if crop.size == 0:
            return False
        avg_color = np.mean(crop.reshape(-1, 3), axis=0)
        return all(avg_color > ONLY_WHITE_THRESHOLD)

    def iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
        boxBArea = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def count_raised_fingers(self, hand_landmarks):
        tips = [4, 8, 12, 16, 20]
        count = 0
        for tip_id in tips:
            if tip_id == 4:
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    count += 1
            else:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    count += 1
        return count

    def track(self):
        ret, frame = self.cap.read()
        if not ret:
            return {"target_locked": False, "center_x": None, "center_y": None}

        frame = cv2.resize(frame, (640, 480))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = self.hands.process(img_rgb)
        ten_fingers_detected = False
        if results_hands.multi_hand_landmarks:
            total_fingers = 0
            for hand_landmarks in results_hands.multi_hand_landmarks:
                total_fingers += self.count_raised_fingers(hand_landmarks)
            if total_fingers >= 10:
                ten_fingers_detected = True

        results = self.model(img_rgb)
        detections = results.xyxy[0]

        best_match = None
        best_score = 0

        for *xyxy, conf, cls in detections:
            cls = int(cls.item()) if hasattr(cls, 'item') else int(cls)
            label = self.model.names[cls]
            if label != "person":
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            person_crop = frame[y1:y2, x1:x2]

            if not self.target_locked:
                if self.is_wearing_white(person_crop) or ten_fingers_detected:
                    self.target_box = (x1, y1, x2, y2)
                    self.target_locked = True
                    break
            else:
                score = self.iou((x1, y1, x2, y2), self.target_box)
                if score > best_score:
                    best_score = score
                    best_match = (x1, y1, x2, y2)

        if self.target_locked and best_match:
            self.target_box = best_match
            x1, y1, x2, y2 = self.target_box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            return {"target_locked": True, "center_x": center_x, "center_y": center_y}

        return {"target_locked": self.target_locked, "center_x": None, "center_y": None}

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
