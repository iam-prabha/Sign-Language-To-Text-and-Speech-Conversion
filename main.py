import os
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
from cvzone.HandTrackingModule import HandDetector

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHITE_IMAGE_PATH = os.path.join(BASE_DIR, "white.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
REAL_DATASET_DIR = os.path.join(BASE_DIR, "AtoZ")

def load_real_dataset():
    """Load and preprocess the real dataset from the AtoZ folder."""
    X = []  # Images
    y = []  # Labels
    label_map = {chr(65 + i): i for i in range(26)}  # A=0, B=1, ..., Z=25

    # Iterate through each letter folder
    for label in os.listdir(REAL_DATASET_DIR):
        label_dir = os.path.join(REAL_DATASET_DIR, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to 28x28 to match model input
                    img = cv2.resize(img, (28, 28))
                    # Append to data list
                    X.append(img)
                    # Append corresponding label
                    y.append(label_map[label])
                else:
                    print(f"Failed to load image: {img_path}")

    # Convert to numpy arrays and preprocess
    X = np.array(X).reshape(-1, 28, 28, 1) / 255.0  # Normalize to [0, 1]
    y = to_categorical(y, 26)  # One-hot encode labels (26 classes)
    return X, y

def setup_files():
    """Set up white image and train/save the model if it doesn't exist."""
    if not os.path.exists(WHITE_IMAGE_PATH):
        white = np.ones((400, 400), np.uint8) * 255
        cv2.imwrite(WHITE_IMAGE_PATH, white)
        print(f"Created {WHITE_IMAGE_PATH}")

    if not os.path.exists(MODEL_PATH):
        # Load real dataset
        X, y = load_real_dataset()
        if len(X) == 0:
            print("No images found in AtoZ folder. Please check your dataset.")
            return

        # Print dataset shape for verification
        print(f"Dataset shape: X={X.shape}, y={y.shape}")

        # Define the CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(26, activation='softmax')  # 26 classes for A-Z
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(X, y,
                           epochs=10,  # Adjust epochs as needed
                           batch_size=32,
                           validation_split=0.2,  # 20% of data for validation
                           verbose=1)

        # Save the trained model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH} with size {os.path.getsize(MODEL_PATH) / 1024:.2f} KB")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

class SignLanguageApp:
    def __init__(self):
        setup_files()
        self.capture = cv2.VideoCapture(0)
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.model = load_model(MODEL_PATH)
        self.speak_engine = pyttsx3.init()
        self.text = ""
        self.root = tk.Tk()
        self.root.title("Sign Language Converter")
        self.root.geometry("800x600")
        self.video_label = tk.Label(self.root)
        self.video_label.place(x=400, y=0, width=400, height=400)
        self.hand_label = tk.Label(self.root)
        self.hand_label.place(x=0, y=0, width=400, height=400)
        self.load_hand_image()
        self.text_label = tk.Label(self.root, text="Text: ", font=("Arial", 14))
        self.text_label.place(x=0, y=450)
        self.text_display = tk.Label(self.root, text="", font=("Arial", 12))
        self.text_display.place(x=0, y=480)
        self.speak_button = tk.Button(self.root, text="Speak", command=self.speak_text)
        self.speak_button.place(x=600, y=450)
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_text)
        self.clear_button.place(x=600, y=500)
        self.video_loop()

    def load_hand_image(self):
        img = cv2.imread(WHITE_IMAGE_PATH)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.hand_label.imgtk = imgtk
        self.hand_label.configure(image=imgtk)

    def video_loop(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            hands = self.hd.findHands(frame, draw=False)
            if hands:
                hand = hands[0]
                if isinstance(hand, dict) and 'bbox' in hand:
                    x, y, w, h = hand['bbox']
                    image = frame[max(0, y-30):y+h+30, max(0, x-30):x+w+30]
                    white = cv2.imread(WHITE_IMAGE_PATH)
                    handz = self.hd2.findHands(image, draw=False)
                    if handz:
                        hand = handz[0]
                        pts = hand['lmList']
                        os, os1 = ((400 - w) // 2) - 15, ((400 - h) // 2) - 15
                        for t in [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]:
                            for i in range(t[0], t[1]):
                                cv2.line(white, (pts[i][0] + os, pts[i][1] + os1),
                                        (pts[i + 1][0] + os, pts[i + 1][1] + os1), (0, 255, 0), 3)
                        for pair in [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]:
                            cv2.line(white, (pts[pair[0]][0] + os, pts[pair[0]][1] + os1),
                                    (pts[pair[1]][0] + os, pts[pair[1]][1] + os1), (0, 255, 0), 3)
                        img_input = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
                        img_input = cv2.resize(img_input, (28, 28)).reshape(1, 28, 28, 1) / 255.0
                        prediction = self.model.predict(img_input)
                        char_index = np.argmax(prediction)
                        current_char = chr(65 + char_index)
                        if current_char not in self.text:
                            self.text += current_char
                        self.text_display.config(text=self.text)
                        hand_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                        hand_img = Image.fromarray(hand_rgb)
                        handtk = ImageTk.PhotoImage(image=hand_img)
                        self.hand_label.imgtk = handtk
                        self.hand_label.configure(image=handtk)
                else:
                    print("Unexpected hand format:", hand)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.video_loop)

    def speak_text(self):
        self.speak_engine.say(self.text)
        self.speak_engine.runAndWait()

    def clear_text(self):
        self.text = ""
        self.text_display.config(text=self.text)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()