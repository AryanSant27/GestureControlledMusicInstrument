import os
import mediapipe as mp
import cv2
import pygame
import streamlit as st
from PIL import Image
import numpy as np
import time
import json
import re
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def load_key_mapping(file_path="key_mapping.json"):
    if not os.path.exists(file_path):
        return {"Piano": {},"Drums": {},"DrumNames": {}, "PianoNames": {}}  # Return empty dictionary if file doesn't exist

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        # Convert string gesture keys back to integers
        data["Piano"] = {int(k): v for k, v in data["Piano"].items()}
        data["Drums"] = {int(k): v for k, v in data["Drums"].items()} 
        data["DrumNames"] = {int(k): v for k, v in data["DrumNames"].items()}
        data["PianoNames"] = {int(k): v for k, v in data["PianoNames"].items()}
        return data
    except json.JSONDecodeError:
        print("Error: Corrupt JSON file. Resetting to empty mapping.")
        return {"Piano": {}, "Drums": {}, "DrumNames": {}, "PianoNames": {}}    # Return empty mapping on error

def save_key_mapping(mapping, file_path="key_mapping.json"):
    
    # Load the existing file to keep 'DrumNames'
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {"DrumNames": {}, "PianoNames": {}}  # Ensure it has DrumNames if file is missing

    # Preserve DrumNames while updating Piano & Drums
    existing_data.update({
        "Piano": {str(k): v for k, v in mapping["Piano"].items()},
        "Drums": {str(k): v for k, v in mapping["Drums"].items()}
    })

    # Save back to JSON
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

def gesture_to_fingers(gesture):
    """Convert integer gesture key to finger combination."""
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    active_fingers = [fingers[i] for i in range(5) if (gesture & (1 << i)) > 0]
    return " ".join(active_fingers) if active_fingers else "None"

def generate_cheat_sheet(file_path="drum_cheat_sheet.txt"):
    """Generate a cheat sheet mapping gestures to drum names with correct finger mappings."""
    cheat_lines = []

    for gesture, drum_path in key_mapping["Drums"].items():
        gesture = int(gesture)  # Convert string keys to integers

        # Extract correct drum number from filename (e.g., "drum03.mp3" → 3)
        match = re.search(r"drum(\d+)", drum_path)
        if match:
            drum_number = int(match.group(1))  # Convert extracted string "03" to integer 3
            drum_name = key_mapping["DrumNames"].get(drum_number, f"Drum {drum_number}")
        else:
            drum_name = f"Drum {gesture}"  # Fallback if extraction fails

        finger_combination = gesture_to_fingers(gesture)
        cheat_lines.append(f"{drum_name} - {finger_combination}")

    # Save to a .txt file
    with open(file_path, "w") as f:
        f.write("\n".join(cheat_lines))

    print(f"Cheat sheet saved to {file_path}")
    
def generate_piano_cheat_sheet(file_path="piano_cheat_sheet.txt"):
    """Generate a cheat sheet mapping gestures to piano key names with correct finger mappings."""
    cheat_lines = []

    for gesture, piano_path in key_mapping["Piano"].items():
        gesture = int(gesture)  # Convert string keys to integers

        # Extract correct piano key number from filename (e.g., "key03.ogg" → 3)
        match = re.search(r"key(\d+)", piano_path)
        if match:
            key_number = int(match.group(1))  # Convert extracted string "03" to integer 3
            key_name = key_mapping["PianoNames"].get(key_number, f"Key {key_number}")
        else:
            key_name = f"Key {gesture}"  # Fallback if extraction fails

        finger_combination = gesture_to_fingers(gesture)
        cheat_lines.append(f"{key_name} - {finger_combination}")

    # Save to a .txt file
    with open(file_path, "w") as f:
        f.write("\n".join(cheat_lines))

    return file_path  # Return the path for download

def run_camera(instrument):
    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.set_num_channels(10)  # Multiple audio channels
    # Initialize MediaPipe Hands
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    # Video Capture
    cap = cv2.VideoCapture(0)

    last_played = {}
    channels = [pygame.mixer.Channel(i) for i in range(10)]
    last_hit = {}  # Track last played drum or piano key
    active_sound = None  # Highlighted sound
    hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.5)
    if st.session_state.start:

        stframe = st.empty()
        
        frame_skip = 1  # Initial frame skipping value
        prev_time = time.time()
        frame_count = 0

        while cap.isOpened() and st.session_state.start:
            sound_mapping = instruments[instrument]
            success, frame = cap.read()
            if not success:
                st.warning("Could not access webcam.")
                break
            
            frame_count += 1

            # Measure processing time
            current_time = time.time()
            elapsed_time = current_time - prev_time

            # Adjust frame skipping dynamically based on processing speed
            if elapsed_time > 0.03:  # If processing slows down (target ~30 FPS)
                frame_skip = min(frame_skip + 1, 5)  # Increase skipping (Max 5)
            else:
                frame_skip = max(frame_skip - 1, 1)  # Decrease skipping (Min 1)

            prev_time = current_time

            # Skip frames dynamically
            if frame_count % frame_skip != 0:
                continue  # Skip processing this frame
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(hand_landmarks.landmark)}

                    # Detect individual fingers (binary mapping)
                    fingers_up = {
                        "thumb": landmarks[4][0] > landmarks[3][0],  # Thumb outward
                        "index": landmarks[8][1] < landmarks[6][1],  # Index raised
                        "middle": landmarks[12][1] < landmarks[10][1],  # Middle raised
                        "ring": landmarks[16][1] < landmarks[14][1],  # Ring raised
                        "pinky": landmarks[20][1] < landmarks[18][1],  # Pinky raised
                    }

                    # Remove standalone ring finger detection
                    if fingers_up["ring"] and not (fingers_up["index"] or fingers_up["middle"] or fingers_up["pinky"]):
                        fingers_up["ring"] = False  # Only count ring if part of a combination

                    # Convert finger positions to a binary number
                    fingers_binary = (
                                        fingers_up["thumb"] * 0b00001 +
                                        fingers_up["index"] * 0b00010 +
                                        fingers_up["middle"] * 0b00100 +
                                        fingers_up["ring"] * 0b01000 +
                                        fingers_up["pinky"] * 0b10000
                                    )

                    # Initialize previous gesture outside the loop
                    if "prev_gesture" not in locals():
                        prev_gesture = -1  # Default value, ensuring first gesture is recognized

                    # Check if hand is fully closed (all fingers down)
                    hand_closed = fingers_binary == 0  # Fist condition

                    # Reset previous gesture when hand is fully closed
                    if hand_closed:
                        prev_gesture = -1  # Reset so the same key can be played again

                    # Playing Piano
                    if instrument == "Piano":
                        if fingers_binary in instruments["Piano"]:
                            sound_path = instruments["Piano"].get(fingers_binary)

                            # Play only if it's a new gesture after reset (no cooldown)
                            if sound_path and os.path.exists(sound_path) and fingers_binary != prev_gesture:
                                free_channel = next((ch for ch in channels if not ch.get_busy()), None)
                                if free_channel:
                                    free_channel.play(pygame.mixer.Sound(sound_path))
                                    active_sound = fingers_binary
                            prev_gesture = fingers_binary  # Update gesture tracking

                    # Playing Drums
                    if instrument == "Drums":
                        sound_path = instruments["Drums"].get(fingers_binary)

                        if sound_path and os.path.exists(sound_path) and fingers_binary != prev_gesture:
                            try:
                                drum_sound = pygame.mixer.Sound(sound_path)
                                free_channel = next((ch for ch in channels if not ch.get_busy()), None)
                                if free_channel:
                                    free_channel.play(drum_sound)
                                last_hit[fingers_binary] = time.time()
                                active_sound = fingers_binary
                                prev_gesture = fingers_binary
                            except pygame.error as e:
                                print(f"Error loading drum sound: {e}")


        # Draw Animation with Text for Both Drums and Piano
            if active_sound is not None and time.time() - last_hit.get(active_sound, 0) < 0.5:
                sound_pos = (320, 400)  # Fixed position for animation
                cv2.circle(frame, sound_pos, 50, (0, 255, 0), -1)  # Draw active sound in green

                label = ""

                if instrument == "Piano":
                    # Reverse lookup: Get the mapped key number
                    key_number = None
                    for gesture, sound_path in instruments["Piano"].items():
                        if gesture == active_sound:
                            key_number = sound_path.split("key")[1].split(".")[0]  # Extract "02" from "key02.ogg"
                            break

                    label = f"Key-{key_number}" if key_number else f"Key-{active_sound}"  # Show key if found
                else:
                    label = f"Drum-{active_sound}"  # Drums still display gesture code

                cv2.putText(frame, label, (sound_pos[0] - 30, sound_pos[1] + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Add text inside circle

            img = Image.fromarray(frame)
            stframe.image(img, channels="RGB")

        # Update Video Feed
        img1 = Image.fromarray(frame)
        video_placeholder.image(img1, channels="RGB")

        cap.release()

def customize_hand_gesture(instrument):
    st.subheader(f"Customize {instrument} Hand Gestures")

        # Create two columns: Left for checkboxes, Right for image
    col1, col2 = st.columns([1, 1])  # Adjust width ratio if needed

    with col1:
            # Checkbox inputs for finger selection
            thumb = st.checkbox("Thumb")
            index = st.checkbox("Index")
            middle = st.checkbox("Middle")
            ring = st.checkbox("Ring")
            pinky = st.checkbox("Pinky")

        # Define the gesture image path
    IMAGE_FOLDER = "C:/Users/hp/Desktop/GestureControlledMusicPlayer/images/Hand-images/"
    gesture_images = {i: os.path.join(IMAGE_FOLDER, f"{i}.png") for i in range(32)}

    # Convert selected fingers to binary code
    custom_gesture = (
            thumb * 0b00001 +
            index * 0b00010 +
            middle * 0b00100 +
            ring * 0b01000 +
            pinky * 0b10000
        )

        # Store gesture selection in session state
    st.session_state.selected_gesture = custom_gesture

    with col2:
            # Display the corresponding gesture image 
        if custom_gesture in gesture_images:
            gesture_path = gesture_images[custom_gesture]

                # Convert to base64 for displaying
            gesture_base64 = get_base64_image(gesture_path)

            # Adjust vertical alignment using CSS
            st.markdown(
                    f"""
                    <div style="margin-top: -80px;"> <!-- Adjust this value -->
                        <img src="data:image/png;base64,{gesture_base64}" width="300">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No gesture image found for this selection.")

        # Dropdown for selecting key mapping (update dynamically)
    if instrument == "Piano":
        key_options = list(key_mapping["PianoNames"].values())
    elif instrument == "Drums":
        key_options = [key_mapping["DrumNames"].get(i, f"Drum {i}") for i in range(1, 11)]  # Fetch names from JSON

    selected_key = st.selectbox("Assign to", key_options)

    if "error_message" not in st.session_state:
        st.session_state.error_message = None  # Initialize session state for error message

        # Button to save custom mapping
    if st.button("Save"):
        custom_gesture = st.session_state.get("selected_gesture", None)  # Ensure it exists
            # Prevent assigning a closed hand to any key
        if custom_gesture == 0:
            st.session_state.error_message = "⚠️ Error: Cannot assign a key to a closed hand gesture! Please use at least one open finger."
            st.rerun()
        else:
            st.session_state.error_message = None  # Reset error if gesture is valid
            custom_gesture = st.session_state.selected_gesture  # Use existing value
            if instrument == "Piano":
                key_number = next((k for k, v in key_mapping["PianoNames"].items() if v == selected_key), None)  # Reverse lookup
            else:  # Drums
                key_number = next((k for k, v in key_mapping["DrumNames"].items() if v == selected_key), None)  # Reverse lookup

            if instrument == "Piano":
                new_sound_path = f"C:/Users/hp/Desktop/GestureControlledMusicPlayer/sounds/Piano/key{key_number:02}.ogg"
                mapping_dict = instruments["Piano"]
            else:  # Drums
                if key_number is not None:
                    new_sound_path = f"C:/Users/hp/Desktop/GestureControlledMusicPlayer/sounds/Drums/drum{key_number:02}.mp3"
                    mapping_dict = instruments["Drums"]

                # Find the old gesture assigned to this key
            old_gesture = None
            for gesture, path in instruments[instrument].items():
                if path == new_sound_path:
                    old_gesture = gesture
                    break

                # Find if the new gesture was already assigned to another key
            prev_key_for_custom_gesture = None
            for key, path in instruments[instrument].items():
                if path == instruments[instrument].get(custom_gesture):  
                    prev_key_for_custom_gesture = key
                    break

            if old_gesture:
                if prev_key_for_custom_gesture:
                    # Swap the gestures correctly without losing assignments
                    instruments[instrument][old_gesture], instruments[instrument][prev_key_for_custom_gesture] = (
                        instruments[instrument][prev_key_for_custom_gesture],
                        instruments[instrument][old_gesture]
                    )

                else:
                    # If the new gesture was not mapped before, remove the old gesture and assign the new one
                    instruments[instrument].pop(old_gesture, None)  # Remove safely
                    instruments[instrument][custom_gesture] = new_sound_path

            else:
                # If there was no old gesture, just assign the new gesture
                instruments[instrument][custom_gesture] = new_sound_path

            # Save changes to the JSON file  
            save_key_mapping(instruments)
            st.session_state.success_message = "✅ Saved!"
        st.rerun()
    # Display error message persistently after rerun
    if st.session_state.get("error_message"):
        st.error(st.session_state.error_message)
        # Clear the message after displaying it
        st.session_state.error_message = None

    if st.session_state.get("success_message"):
        st.success(st.session_state.success_message)
        # Clear the message after displaying it
        st.session_state.success_message = None

def create_download_button(instrument):
    """Creates a download button for the selected instrument's cheat sheet."""
    
    if instrument == "Drums": 
        cheat_sheet_text = generate_cheat_sheet()
        file_path = "drum_cheat_sheet.txt"

    elif instrument == "Piano":  
        file_path = generate_piano_cheat_sheet()

    else:
        return  # Exit if the instrument is not valid

    # Ensure file exists before reading
    try:
        with open(file_path, "rb") as f:
            cheat_sheet_data = f.read()

        # Create download button
        st.download_button(
            label=f"📥 {instrument} Controls",
            data=cheat_sheet_data,
            file_name=file_path,
            mime="text/plain"
        )

    except FileNotFoundError:
        st.error(f"{instrument} cheat sheet file not found. Please generate it first.")
    except Exception as e:
        st.error(f"Error reading file: {e}")    

def display_instrument_selection():
    """Displays instrument selection buttons and handles URL parameters."""
    
    if st.session_state.instrument is None:
        # Encode images
        drums_base64 = get_base64_image("C:/Users/hp/Desktop/GestureControlledMusicPlayer/images/drums-icon.png")
        piano_base64 = get_base64_image("C:/Users/hp/Desktop/GestureControlledMusicPlayer/images/piano-icon.png")

        with st.container():
            st.markdown(
                """
                <div style="text-align: center;">
                    <h1 style="font-size: 40px; font-weight: bold;">Wyric.io</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Centered Instrument Images
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; gap: 50px; margin-top: 20px;">
                    <a href="?instrument=Drums">
                        <img src="data:image/png;base64,{drums_base64}" width="275">
                    </a>
                    <a href="?instrument=Piano">
                        <img src="data:image/png;base64,{piano_base64}" width="275">
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Centered "Choose an Instrument" text
            st.markdown(
                """
                <div style="text-align: center; margin-top: 20px; font-size: 24px; font-weight: bold;">
                    Choose an Instrument 
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Handle URL query params
        if "instrument" in st.query_params:
            st.session_state.instrument = st.query_params["instrument"]
            st.query_params["scroll"] = "top"
            st.rerun()

        st.stop()  # Stop execution so only buttons appear

# Load key mapping
key_mapping = load_key_mapping()
# Define sound mappings for different instruments
instruments = {
    "Drums": key_mapping["Drums"], # Load mappings from the JSON file
    "Piano": key_mapping["Piano"]  # Load mappings from the JSON file
}
# Initialize session state for instrument selection
if "instrument" not in st.session_state:
    st.session_state.instrument = None  # No instrument selected initially

# Streamlit UI

# Display notification only once
if "not_first" not in st.session_state:
    st.toast("Best experience in dark mode", icon="✊🏿")
    st.session_state.not_first = False
    
display_instrument_selection()

instrument=st.session_state.instrument

if instrument == "Drums":
    title_text = "THIS APP MAKES YOU PLAY BETTER DRUMS THAN RINGO STARR FROM THE BEATLES"
else:
    title_text = "BEETHOVEN IS CRYING BLOOD TEARS IN HEAVEN AFTER SEEING THIS APP"

marquee_html = f"""
<div style="white-space: nowrap; overflow: hidden;">
    <marquee style="font-size:24px; font-weight:bold; color:red;">{title_text}</marquee>
</div>
"""
st.markdown(marquee_html, unsafe_allow_html=True)

if instrument:

    create_download_button(instrument)
    
    # Initialize Session State
    if "start" not in st.session_state:
        st.session_state.start = False
    if "custom_gesture" not in st.session_state:
        st.session_state.custom_gesture = None  # Store selected gesture
    if "error_message" not in st.session_state:
        st.session_state.error_message = None
    if "success_message" not in st.session_state:
        st.session_state.success_message = None

    # Apply custom CSS
    st.markdown(
        """
        <style>
            .stButton>button {
                background-color: red !important;
                color: white;
            }
            .stButton>button:hover {
                background-color: white !important;
                color: red !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Centered Play Button
    col1, col2, col3 = st.columns([1, 2, 1])  # Center align
    with col2:
        if st.button("PLAY", key="play-btn", use_container_width=True):
            st.session_state.start = True
            st.rerun()

# Hand gesture customization
st.markdown('<div id="customization-section"></div>', unsafe_allow_html=True)
customize_hand_gesture(instrument)

# Placeholder for Video Feed
st.markdown('<div id="video-section"></div>', unsafe_allow_html=True)
video_placeholder = st.empty()

# Start Video 
if st.session_state.start:
     run_camera(instrument)
