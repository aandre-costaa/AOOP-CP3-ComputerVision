import cv2
import numpy as np
import string
import time
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model("sign_lang_model.keras")
classes = list(string.digits + string.ascii_uppercase)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

top_left = (50, 50)
bottom_right = (350, 350)

cap = cv2.VideoCapture(0)

# Word creator logic
prev_letter = ''
last_detection_time = 0
cooldown = 1.5
phrase = ''

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    black = np.zeros_like(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    letter = "-"
    confidence = 0

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(black, landmarks, mp_hands.HAND_CONNECTIONS)

        # Crop and prepare input
        roi = black[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        resized = cv2.resize(gray, (64, 64))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 64, 64, 1)

        prediction = model.predict(reshaped, verbose=0)
        letter = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Add letter if it's new or different from the last and cooldown passed
        current_time = time.time()
        if confidence > 85 and letter != prev_letter and current_time - last_detection_time > cooldown:
            phrase += letter
            prev_letter = letter
            last_detection_time = current_time

    cv2.putText(frame, f"Letter: {letter} ({confidence:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(frame, f"{phrase}", (10, 400),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

    cv2.imshow("Sign Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # press 'c' to clear the word
        phrase = ''
        prev_letter = ''
    elif key == ord(' '):
        phrase += ' '
    elif key == 8: #Backspace
        phrase = phrase[:-1]
cap.release()
cv2.destroyAllWindows()
