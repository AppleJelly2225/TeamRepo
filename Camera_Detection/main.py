import sys
import platform
import time
import cv2
import numpy as np
import torch
import socket

# Below is configuration
USE_DISPLAY = True                   # Display the video window
SEND_NETWORK = False                 # Maybe could send detection results to a remote ground station?
REMOTE_IP   = "192.168.1.100"        # Remote ground station IP (change to whatever it actually is)
REMOTE_PORT = 5005                   # Port number for sending detection data?
USE_DRONEKIT = False                 # Send alerts to Pixhawk via DroneKit MAVLink? Not sure how to do it.
PIXHAWK_CONNECTION = "/dev/ttyACM0"  # Connection string for Pixhawk (serial port or IP address)

# Place droneKit to enabled feature, import dronekit and connect to the vehicle.
vehicle = None
if USE_DRONEKIT:
    try:
        from dronekit import connect, VehicleMode
    except ImportError:
        print("Dronekit not installed. Need to install: pip install dronekit")
        sys.exit(1)
    # Connect to Pixhawk. Adjust baudrate if needed. Not sure how.
    print(f"Connecting to Pixhawk at {PIXHAWK_CONNECTION}...")
    vehicle = connect(PIXHAWK_CONNECTION, wait_ready=True, baud=115200)
    # 'Vehicle' is DroneKit vehicle instance (if connection succeeds).

# Set up socket for network sending (if enabled)
sock = None
if SEND_NETWORK:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Sending detection results to {REMOTE_IP}:{REMOTE_PORT} via UDP...")

# Pick a device: use GPU if available or CPU if not.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Loading YOLOv5 model (yolov5n - nano model for speed on Pi, or we can use the biggest one, yolov5x ) >>> smaller model to improve Pi performance, hopefully it helps
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, device=device)
model.conf = 0.25  # Confidence limit/threshold for predictions
# The model detects all 80 COCO (a dataset) classes (objects).

# Initialize webcam video capture
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    # Try other camera if first one is not working
    for idx in range(1, 4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Using a new camera index {idx}")
            break
if not cap.isOpened():
    print("Error: Fail to open the camera.")
    sys.exit(1)

# Maybe reduce resolution to improve speed (for Pi)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Start the video. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame (BGR OpenCV image) to RGB because YOLOv5 expects RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLOv5 model
    results = model(img)

    # Shows detection results. `results.xyxy[0]` gives [[x1,y1,x2,y2,conf,class], ...] for the frame
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        # *xyxy captures x1,y1,x2,y2, then conf, cls are last two.
        class_id = int(cls.item()) if hasattr(cls, 'item') else int(cls)        # get the class index as int
        conf_val = float(conf.item()) if hasattr(conf, 'item') else float(conf)
        class_name = model.names[class_id]                                      # Get class (object) name from model's names list
        detections.append((class_name, conf_val, xyxy))

        # Draw a box and label on the frame
        if USE_DISPLAY:
            x1, y1, x2, y2 = map(int, xyxy)  # convert coordinates to int
            label = f"{class_name} {conf_val:.2f}"
            # Draw another box and label on its original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2, cv2.LINE_AA)

    # Sending to remote ground station and the detection data
    if SEND_NETWORK and detections:
        # A simple message
        detected_names = [det[0] for det in detections]
        message = f"Detected: {', '.join(detected_names)}"
        try:
            sock.sendto(message.encode('utf-8'), (REMOTE_IP, REMOTE_PORT))
        except Exception as e:
            print(f"UDP send error: {e}")

    # If using DroneKit/Pixhawk integration, send MAVLink message on detections (not sure how)
    if USE_DRONEKIT and vehicle is not None and detections:
        # Show a message with detected object name
        detected_names = [det[0] for det in detections]
        text_msg = ("Objects: " + ", ".join(detected_names))[:50]  # MAVLink STATUSTEXT max 50 chars (this is what I googled but not sure why)
        msg = vehicle.message_factory.statustext_encode(
            6,                  # severity
            text_msg.encode()   # text, need to be byte-encoded
        )
        # Send the message via DroneKit
        vehicle.send_mavlink(msg)
        vehicle.flush()         # Flush so it could immediately send

    # Show the video frame with detections
    if USE_DISPLAY:
        cv2.imshow("YOLOv5 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Keep trying
    if not USE_DISPLAY:
        time.sleep(0.01)  # Small delay to GPU

# Cleanup
cap.release()
if USE_DISPLAY:
    cv2.destroyAllWindows()
if USE_DRONEKIT and vehicle is not None:
    vehicle.close()
    print("Drone connection closed.")
print("Program Ends.")
