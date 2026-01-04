import cv2
from ultralytics import YOLO
import pygame
import threading
import time

# ================== SOUND SETUP ==================
pygame.mixer.init()
pygame.mixer.music.load("siren.mp3")

siren_playing = False

def play_siren():
    global siren_playing
    if not siren_playing:
        siren_playing = True
        pygame.mixer.music.play(-1)  # loop

def stop_siren():
    global siren_playing
    if siren_playing:
        pygame.mixer.music.stop()
        siren_playing = False

# ================== YOLO MODEL ==================
model = YOLO("yolov8n.pt")

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera not found")
    exit()

# ================== ROI (ANALYSIS ZONE) ==================
ROI_X1, ROI_Y1 = 350, 180
ROI_X2, ROI_Y2 = 930, 620

# ================== STABILITY CONTROL ==================
DETECTION_FRAMES_REQUIRED = 8     # phone must stay for these frames
NO_DETECTION_FRAMES = 10          # stop siren after this

detect_counter = 0
lost_counter = 0

print("System running | Press Q to quit")

# ================== CAMERA LENS DETECTOR ==================
def detect_camera_lens(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=120,
        param2=35,
        minRadius=6,
        maxRadius=30
    )
    return circles is not None

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
    cv2.putText(frame, "ANALYSIS ZONE", (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    phone_detected = False

    # -------- YOLO DETECTION (PHONE BODY) --------
    results = model(roi, conf=0.25, verbose=False)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 67:  # cell phone
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(
                    frame,
                    (x1 + ROI_X1, y1 + ROI_Y1),
                    (x2 + ROI_X1, y2 + ROI_Y1),
                    (0, 0, 255), 2
                )

    # -------- CAMERA LENS DETECTION --------
    lens_detected = detect_camera_lens(roi)

    detection = phone_detected or lens_detected

    # -------- STABILITY LOGIC --------
    if detection:
        detect_counter += 1
        lost_counter = 0
    else:
        lost_counter += 1
        detect_counter = 0

    # Start siren only after stable detection
    if detect_counter >= DETECTION_FRAMES_REQUIRED:
        threading.Thread(target=play_siren, daemon=True).start()

    # Stop siren only after object disappears
    if lost_counter >= NO_DETECTION_FRAMES:
        stop_siren()

    # Status text
    status = "DETECTING" if detection else "CLEAR"
    color = (0, 0, 255) if detection else (0, 255, 0)
    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Phone Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== CLEANUP ==================
stop_siren()
cap.release()
cv2.destroyAllWindows()
