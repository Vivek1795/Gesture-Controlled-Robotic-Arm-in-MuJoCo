import cv2
import csv
import mediapipe as mp
import numpy as np
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# CSV file
csv_filename = "augmented_gesture_data.csv"
header_written = False

# Define gesture labels
gesture_labels = {
    0: "fist",
    1: "open_palm",
    2: "left",
    3: "right",
    4: "thumbs_up"
}

print("Press keys [0–4] to label gestures, 's' to save current, 'q' to quit.")

cap = cv2.VideoCapture(0)
current_label = None
saved_count = 0

with open(csv_filename, mode="a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                if current_label is not None:
                    cv2.putText(image_bgr, f"Label: {gesture_labels[current_label]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Save sample
                    writer.writerow(features + [current_label])
                    saved_count += 1
                    print(f"Saved sample #{saved_count} as {gesture_labels[current_label]}")
                    current_label = None

        cv2.imshow("Collect Gesture Data", image_bgr)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key in [ord(str(k)) for k in gesture_labels.keys()]:
            current_label = int(chr(key))

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Finished. Total saved samples: {saved_count}")