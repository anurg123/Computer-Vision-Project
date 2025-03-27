# Rock Paper Scissors - Computer Vision Project

## 📌 Project Overview
This project is a computer vision-based Rock Paper Scissors game that uses a webcam to detect hand gestures and determine whether the user has shown Rock, Paper, or Scissors. It is implemented using Python with OpenCV and MediaPipe for real-time hand tracking and gesture recognition.

## 🎯 Features
- Real-time hand gesture detection using OpenCV and MediaPipe.
- Classification of Rock, Paper, and Scissors using finger landmark positions.
- Simple GUI to display user and computer choices.
- Random AI opponent to play against.
- Score tracking for multiple rounds.

## 🛠️ Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
- Random (for AI opponent)

---

## 📌 Installation & Setup

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/yourusername/rock-paper-scissors-cv.git
   cd rock-paper-scissors-cv
   ```
2. **Install Dependencies:**
   ```sh
   pip install opencv-python mediapipe numpy
   ```
3. **Run the Game:**
   ```sh
   python rock_paper_scissors.py
   ```

---

## 📌 Implementation Details

### 1️⃣ Importing Required Libraries
```python
import cv2
import mediapipe as mp
import numpy as np
import random
```

### 2️⃣ Initializing MediaPipe Hand Tracking
```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
```

### 3️⃣ Capturing Webcam Input
```python
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
```

### 4️⃣ Detecting Hand Gestures
```python
def detect_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, and Pinky fingertips
    thumb_tip = 4
    
    fingers = []
    for tip in finger_tips:
        fingers.append(landmarks[tip][1] < landmarks[tip - 2][1])
    
    if all(fingers):
        return "Paper"
    elif fingers.count(True) == 0:
        return "Rock"
    elif fingers[0] and not any(fingers[1:]):
        return "Scissors"
    return "Unknown"
```

### 5️⃣ Game Logic & Display
```python
computer_choices = ["Rock", "Paper", "Scissors"]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            user_choice = detect_gesture(landmarks)
            comp_choice = random.choice(computer_choices)
            cv2.putText(image, f"You: {user_choice} | AI: {comp_choice}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Rock Paper Scissors", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## 📌 How to Play
1. Start the game by running `rock_paper_scissors.py`.
2. Show Rock ✊, Paper ✋, or Scissors ✌️ in front of your webcam.
3. The AI opponent randomly selects a move.
4. The game displays both choices and determines the winner.
5. Press `q` to quit the game.

---

## 📌 Future Improvements
- Implement a trained deep-learning model for gesture classification.
- Enhance UI with better visuals.
- Add multiplayer mode using sockets.

---

## 📌 Author
**Ananya**

---

## 📌 License
This project is open-source and available under the MIT License.
