import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define word pairs in the correct order
word_pairs = [
    ("Date", "Late", 'a', 's'),
    ("Brass", "Grass", 'd', 'f'),
    ("School", "Pool", 'g', 'h'),
    ("Steal", "Meal", 'j', 'k'),
    ("Bad", "Dad", 'z', 'x'),  # McGurk effect
    ("Tin", "Bin", 'n', 'm')
]
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


# Define the relative path to the video folder
video_folder = os.path.join(current_dir, "audio_clips/trial_subject_4")

# Paths to pre-recorded lip videos using relative paths
video_paths = {
    "a": os.path.join(video_folder, "date.mov"),
    "s": os.path.join(video_folder, "late.mov"),
    "d": os.path.join(video_folder, "brass.mov"),
    "f": os.path.join(video_folder, "grass.mov"),
    "g": os.path.join(video_folder, "school.mov"),
    "h": os.path.join(video_folder, "pool.mov"),
    "j": os.path.join(video_folder, "steal.mov"),
    "k": os.path.join(video_folder, "meal.mov"),
    "z": os.path.join(video_folder, "bad.mov"),
    "x": os.path.join(video_folder, "dad.mov"),
    "n": os.path.join(video_folder, "tin.mov"),
    "m": os.path.join(video_folder, "bin.mov"),
}
# Verify video paths
for key, path in video_paths.items():
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.")
        exit()

# Open video captures
video_captures = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
for key, cap in video_captures.items():
    if not cap.isOpened():
        print(f"Error: Could not open video for key '{key}', path: {video_paths[key]}")
        exit()

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# State variables
current_video = None
video_playing = False  # Track whether a video is currently playing
current_pair_index = 0  # Start with the first word pair
static_lip_bbox = None  # Store the bounding box of lips
extension_factor = 1.9  # Extend the overlay slightly beyond bounding box

# Variables for text animation
text_alpha = 0  # Controls text transparency
text_direction = 1  # 1 for fade in, -1 for fade out
animation_speed = 5
current_text = ""

def get_lip_bounding_box(landmarks, w, h):
    """Calculate the bounding box for lips based on face landmarks."""
    outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17]
    lip_points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in outer_lip_indices]

    x_min = min([p[0] for p in lip_points])
    y_min = min([p[1] for p in lip_points])
    x_max = max([p[0] for p in lip_points])
    y_max = max([p[1] for p in lip_points])

    return x_min, y_min, x_max, y_max

def animate_text(frame, text, x, y):
    global text_alpha, text_direction

    # Update alpha value
    text_alpha += animation_speed * text_direction

    # Reverse direction at limits
    if text_alpha >= 255:
        text_alpha = 255
        text_direction = -1
    elif text_alpha <= 0:
        text_alpha = 0
        text_direction = 1

    # Render text with current alpha
    overlay = frame.copy()
    cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, text_alpha), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, text_alpha / 255.0, frame, 1 - text_alpha / 255.0, 0, frame)


while True:
    # Get the current word pair
    word1, word2, key1, key2 = word_pairs[current_pair_index]

    # Create a blank white window for text instructions
    text_window = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(text_window, f"{word1} - press {key1.upper()}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(text_window, f"{word2} - press {key2.upper()}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(text_window, "Press ENTER to go to the next pair", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the text window
    cv2.imshow("Text Instructions", text_window)

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam")
        break

    # Flip the frame to act as a mirror
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find facial landmarks
    results = face_mesh.process(rgb_frame)

    # Update lip bounding box when a video is triggered
    if video_playing and static_lip_bbox is None and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Detect and store the initial bounding box for lips
            x_min, y_min, x_max, y_max = get_lip_bounding_box(face_landmarks.landmark, w, h)
            lip_width = int((x_max - x_min) * extension_factor)
            lip_height = int((y_max - y_min) * extension_factor)
            x_min = max(0, x_min - int((lip_width - (x_max - x_min)) / 2))
            y_min = max(0, y_min - int((lip_height - (y_max - y_min)) / 2))
            x_max = min(w, x_min + lip_width)
            y_max = min(h, y_min + lip_height)
            static_lip_bbox = (x_min, y_min, x_max, y_max)
            print("Lip bounding box detected:", static_lip_bbox)

    # Overlay the selected video onto the static lip bounding box
    if video_playing and current_video and static_lip_bbox:
        x_min, y_min, x_max, y_max = static_lip_bbox
        lip_width = x_max - x_min
        lip_height = y_max - y_min

        ret_video, video_frame = current_video.read()
        if not ret_video:
            print("Video finished, resetting state...")
            video_playing = False
            current_video = None
            static_lip_bbox = None
            continue

        # Resize and overlay the video frame
        if video_frame is not None:
            resized_video_frame = cv2.resize(video_frame, (lip_width, lip_height))
            frame[y_min:y_max, x_min:x_max] = resized_video_frame
        else:
            print("Error: video_frame is None, skipping overlay.")

    # Animate text at the bottom of the screen
    if video_playing and current_text:
        text_size = cv2.getTextSize(current_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (w - text_size[0]) // 2  # Center the text horizontally
        text_y = h - 30  # Position the text at the bottom
        animate_text(frame, current_text, text_x, text_y)

    # Display the frame with the video overlay
    cv2.imshow("Magic Mirror - Lip Overlay", frame)

    # Handle keypress events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the application
        break
    # Handle keypress for playing videos
    elif key == ord(key1) and not video_playing:  # Check if it's the "Bad" and "Dad" case
        if word1 == "Bad" and word2 == "Dad":  # Swap logic
            current_video = video_captures[key2]  # Play "Dad" for "Bad"
            current_text = "Dad"
        else:
            current_video = video_captures[key1]
            current_text = word1
        current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        video_playing = True
        static_lip_bbox = None  # Reset lip detection
        text_alpha = 0  # Reset animation
        text_direction = 1
        print(f"Playing '{word1}' video")

    elif key == ord(key2) and not video_playing:
        if word1 == "Bad" and word2 == "Dad":  # Swap logic
            current_video = video_captures[key1]  # Play "Bad" for "Dad"
            current_text = "Bad"
        else:
            current_video = video_captures[key2]
            current_text = word2
        current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        video_playing = True
        static_lip_bbox = None  # Reset lip detection
        text_alpha = 0  # Reset animation
        text_direction = 1
        print(f"Playing '{word2}' video")
        
    elif key == 13:  # Enter key moves to the next pair
        current_pair_index = (current_pair_index + 1) % len(word_pairs)
        video_playing = False
        static_lip_bbox = None
        print(f"Moved to next pair: {word_pairs[current_pair_index]}")

# Release all resources
cap.release()
for vc in video_captures.values():
    vc.release()
cv2.destroyAllWindows()
