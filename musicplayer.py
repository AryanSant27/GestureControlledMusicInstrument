import cv2
import mediapipe as mp
import pygame
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Initialize pygame mixer for sounds
pygame.mixer.init()

# Define the sound mappings for different instruments
instruments = {
    "Drums": {
        1: "E:\LangChain Project\sounds\sounds_cr78-Bongo High.mp3",
        2: "E:\LangChain Project\sounds\sounds_crash.mp3",
        3: "E:\LangChain Project\sounds\sounds_snare.mp3",
        4: "E:\LangChain Project\sounds\sounds_tom-1.mp3",
        5: "E:\LangChain Project\sounds\sounds_tom-2.mp3",
        6: "E:\LangChain Project\sounds\sounds_tom-3.mp3",
        7: "E:\LangChain Project\sounds\sounds_cr78-Cymbal.mp3",
        8: "E:\LangChain Project\sounds\sounds_cr78-Guiro 1.mp3",
        9: "E:\LangChain Project\sounds\sounds_tempest-HiHat Metal.mp3",
        10: "E:\LangChain Project\sounds\sounds_cr78-Bongo High.mp3"
    },
    "Guitar": {
        1: "E:\LangChain Project\sounds\guitar_string_1.mp3",
        2: "E:\LangChain Project\sounds\guitar_string_2.mp3",
        3: "E:\LangChain Project\sounds\guitar_string_3.mp3",
        4: "E:\LangChain Project\sounds\guitar_string_4.mp3",
        5: "E:\LangChain Project\sounds\guitar_string_5.mp3",
        6: "E:\LangChain Project\sounds\guitar_string_6.mp3"
    }
}

# Initialize MediaPipe Hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to start hand tracking after an instrument is selected
def start_hand_tracking(selected_instrument):
    sounds_mapping = instruments[selected_instrument]
    
    # Set up the Tkinter window for hand tracking
    tracking_window = tk.Toplevel(window)
    tracking_window.title(f"Hand Tracking - {selected_instrument}")
    tracking_window.geometry("800x600")

    # Create a label to display the video feed
    video_label = Label(tracking_window)
    video_label.pack()

    cap = cv2.VideoCapture(0)

    def play_sound(finger_count):
        if finger_count in sounds_mapping:
            sound_file = sounds_mapping[finger_count]
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            pygame.time.delay(120)
            pygame.mixer.music.stop()

    def process_frame():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            return

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use MediaPipe Hands for hand landmark detection
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        finger_count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_index = results.multi_hand_landmarks.index(hand_landmarks)
                hand_label = results.multi_handedness[hand_index].classification[0].label
                hand_landmarks_positions = []

                for landmarks in hand_landmarks.landmark:
                    hand_landmarks_positions.append([landmarks.x, landmarks.y])

                if hand_label == "Left" and hand_landmarks_positions[4][0] > hand_landmarks_positions[3][0]:
                    finger_count += 1
                elif hand_label == "Right" and hand_landmarks_positions[4][0] < hand_landmarks_positions[3][0]:
                    finger_count += 1

                if hand_landmarks_positions[8][1] < hand_landmarks_positions[6][1]:
                    finger_count += 1
                if hand_landmarks_positions[12][1] < hand_landmarks_positions[10][1]:
                    finger_count += 1
                if hand_landmarks_positions[16][1] < hand_landmarks_positions[14][1]:
                    finger_count += 1
                if hand_landmarks_positions[20][1] < hand_landmarks_positions[18][1]:
                    finger_count += 1

                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

            # Play corresponding sound based on the finger count
            play_sound(finger_count)

        # Convert the image to display in the GUI
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.img_tk = img_tk
        video_label.configure(image=img_tk)
        
        # Call this function again after a short delay (looping)
        video_label.after(10, process_frame)

    # Start processing frames
    process_frame()

    # Close the video feed when the window is closed
    tracking_window.protocol("WM_DELETE_WINDOW", lambda: cap.release())

# Set up the main Tkinter window for instrument selection
window = tk.Tk()
window.title("Instrument Selection")
window.geometry("900x400")

# Create buttons for each instrument
Label(window, text="Select an Instrument to Play:").pack(pady=10)

for instrument in instruments.keys():
    Button(window, text=instrument, command=lambda inst=instrument: start_hand_tracking(inst)).pack(pady=5)

# Start the Tkinter GUI loop
window.mainloop()

