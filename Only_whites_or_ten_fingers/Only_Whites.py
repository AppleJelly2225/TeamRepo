# Import required libraries
import cv2
import torch
import numpy as np
import mediapipe as mp
import warnings

# This is NOT a bug — just hides some optional warning messages in the terminal
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration settings
DISPLAY_USE_HEADLESS = True             # Set to False when using on drone (if it is headless)
CONF_THRESHOLD = 0.3                    # Confidence threshold for YOLOv5 model object detection
ONLY_WHITE_THRESHOLD = 180                   # Threshold for detecting "white" color in clothing
KEY_1_UNLOCK = ord('r')                   # Press 'r' to reset tracking
KEY_2_QUIT = ord('q')                     # Press 'q' to quit the program

# Load YOLOv5 model from Ultralytics using torch.hub
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Currently using {device}")
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)    # Can change different models: YOLOv5n, YOLOv5m, YOLOv5l, YOLOv5x
model.to(device)
model.eval()
model.conf = CONF_THRESHOLD

# Set up MediaPipe for hand/finger detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Open the webcam/camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")    # Need to check if camera is connected.
    exit()

# Initial the tracking state
target_box = None       # Not showing box/frame
target_locked = False   # Ofc not lock a target yet

# 1st function to check if a cropped region of a person is mostly white (clothes)
def is_wearing_white(crop):
    if crop.size == 0:
        return False
    avg_color = np.mean(crop.reshape(-1, 3), axis=0)
    return all(avg_color > ONLY_WHITE_THRESHOLD)

# 2nd function to calculate the IoU (Intersection-over-Union) between two boxes/frames
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
    boxBArea = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
    return interArea / float(boxAArea + boxBArea - interArea)

# 3rd function to count the number of raised fingers in a detected hand
def count_raised_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    count = 0
    for tip_id in tips:
        if tip_id == 4:  # Thumb (horizontal direction)
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                count += 1
        else:  # Other fingers (vertical direction)
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                count += 1
    return count

# Start the loop for video processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to reduce processing load (This is kind of infect the performance a lot so be careful when changing it.)
    frame = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1st step: Detect hands and count fingers
    results_hands = hands.process(img_rgb)
    ten_fingers_detected = False
    if results_hands.multi_hand_landmarks:
        total_fingers = 0
        for hand_landmarks in results_hands.multi_hand_landmarks:
            total_fingers += count_raised_fingers(hand_landmarks)
            if DISPLAY_USE_HEADLESS:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if total_fingers >= 10:
            ten_fingers_detected = True

    # 2nd step: Use YOLOv5 model to detect people (a person)
    results = model(img_rgb)
    detections = results.xyxy[0]

    best_match = None
    best_score = 0

    for *xyxy, conf, cls in detections:
        cls = int(cls.item()) if hasattr(cls, 'item') else int(cls)
        label = model.names[cls]
        if label != "person":
            continue  # This program only interested in people (for now)

        x1, y1, x2, y2 = map(int, xyxy)
        person_crop = frame[y1:y2, x1:x2]

        # 3rd step: Lock target if wearing white OR showing 10 fingers (Two options available!)
        if not target_locked:
            if is_wearing_white(person_crop) or ten_fingers_detected:
                target_box = (x1, y1, x2, y2)
                target_locked = True
                print("Target locked.")
                break
        else:
            # 4th step: Track the same person using IoU (Intersection-over-Union
            score = iou((x1, y1, x2, y2), target_box)
            if score > best_score:
                best_score = score
                best_match = (x1, y1, x2, y2)

    # 5th step: Update and display tracking box (easier to observe)
    if target_locked and best_match:
        target_box = best_match
        x1, y1, x2, y2 = target_box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        print(f"Tracking: x1={x1}, y1={y1}, x2={x2}, y2={y2}, center=({center_x},{center_y})")

        if DISPLAY_USE_HEADLESS:
            # Color the frame slightly for better visual
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

            # Draw tracking box and label the target
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, "Target", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 6th step: Display video with status
    if DISPLAY_USE_HEADLESS:
        status_text = "Target locked" if target_locked else "Searching for a target"
        color = (0, 255, 0) if target_locked else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Tracking", frame)

        #  Controls ("q" for quit and "r" for relocked)
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_2_QUIT:
            break
        elif key == KEY_1_UNLOCK:
            print("Target unlocked.")
            target_box = None
            target_locked = False

# Clean up the program
cap.release()
cv2.destroyAllWindows()
