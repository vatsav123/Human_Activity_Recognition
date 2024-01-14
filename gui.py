import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import h5py

# Load your pre-trained model
model = load_model('Conv_LSTM_model___Date_Time_2024_01_12__11_40_54___Loss_0.8707373738288879___Accuracy_0.7329192757606506.h5')

# Constants
IMG_HEIGHT, IMG_WIDTH = 64, 64
SEQUENCE_LENGTH = 30
CLASSES_LIST = ['WalkingWithDog', 'TaiChi', 'Swing', 'HorseRace', 'Kayaking']

class HumanActivityRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Human Activity Recognition App")

        # Create labels and buttons
        self.label = tk.Label(self.master, text="Select Video File:")
        self.label.pack()

        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse)
        self.browse_button.pack()

        self.predict_button = tk.Button(self.master, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.quit_button = tk.Button(self.master, text="Quit", command=self.master.quit)
        self.quit_button.pack()

        # Create canvas for displaying video frames
        self.canvas = tk.Canvas(self.master, width=400, height=300)
        self.canvas.pack()

        # Initialize video capture and frame index
        self.cap = None
        self.frame_index = 0

    def browse(self):
        file_path = filedialog.askopenfilename()
        self.cap = cv2.VideoCapture(file_path)
        self.frame_index = 0
        self.show_frame()

    def show_frame(self):
        _, frame = self.cap.read()

        if frame is not None:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame
            frame = cv2.resize(frame, (400, 300))

            # Convert the frame to ImageTk format
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.img_tk = img_tk

            # Repeat after a delay (e.g., 33 milliseconds for ~30 fps)
            self.master.after(33, self.show_frame)
        else:
            self.cap.release()

    def predict(self):
        if self.cap is not None:
            # Capture frames for the sequence
            sequence_frames = []
            for _ in range(SEQUENCE_LENGTH):
                _, frame = self.cap.read()

                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                    frame = img_to_array(frame)
                    frame = preprocess_input(frame)
                    sequence_frames.append(frame)

            # Check if enough frames are captured for the sequence
            if len(sequence_frames) == SEQUENCE_LENGTH:
                # Convert the list of frames to a 5D array
                sequence_frames = np.expand_dims(sequence_frames, axis=0)

                # Make prediction
                prediction = model.predict(sequence_frames)
                activity_label = CLASSES_LIST[np.argmax(prediction)]

                # Show the prediction result
                result_text = f"Predicted Activity: {activity_label}"
                result_label = tk.Label(self.master, text=result_text)
                result_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = HumanActivityRecognitionApp(root)
    root.mainloop()

