import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Paths to the two pre-recorded lip videos
video_x_path = 'audio_clips/trial subject 3/dad.mov'
video_z_path = 'audio_clips/trial subject 3/bad.mov'

# Open the videos
video_x = cv2.VideoCapture(video_x_path)
video_z = cv2.VideoCapture(video_z_path)

# Open the camera
cap = cv2.VideoCapture(0)

# Flags and variables for managing state
current_video = None
video_playing = False  # Track whether a video is currently playing
static_lip_bbox = None  # Store the bounding box of lips
extension_factor = 1.9  # Extend the overlay slightly beyond bounding box

# Variables for text animation
text_alpha = 0  # Controls text transparency
text_direction = 1  # 1 for fade in, -1 for fade out
animation_speed = 5
current_text = ""

def get_lip_bounding_box(landmarks, w, h):
    # Lip landmark indices for outer lip region
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

    # Create a blank white background for the second window
    text_window = np.ones((200, 400, 3), dtype=np.uint8) * 255

    # Add text to the second window
    cv2.putText(text_window, "Dad - press x", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(text_window, "Bad - press z", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the second window
    cv2.imshow("Text Window", text_window)

    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to act as a mirror
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    h, w, _ = frame.shape

    # Convert the frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find facial landmarks
    results = face_mesh.process(rgb_frame)

    # Update lip bounding box when a video is triggered
    if video_playing and static_lip_bbox is None and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Detect and store the initial bounding box for lips
            x_min, y_min, x_max, y_max = get_lip_bounding_box(face_landmarks.landmark, w, h)
            # Apply the extension factor
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

        # Read the next frame from the selected video
        ret_video, video_frame = current_video.read()
        if not ret_video:
            print("Video finished, resetting state...")
            video_playing = False
            current_video = None
            static_lip_bbox = None  # Reset lip bounding box for the next trigger
            continue

        # Check if video_frame is valid before resizing and overlaying
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

    # Display the frame
    cv2.imshow("Magic Mirror - Static Lip Video Overlay", frame)

    # Handle keypress events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('z') and not video_playing:  # Play video_x
        current_video = video_x
        video_x.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        video_playing = True
        static_lip_bbox = None  # Force detection of lips
        current_text = "dad"
        text_alpha = 0  # Reset animation
        text_direction = 1
        print("Playing video_x")
    elif key == ord('x') and not video_playing:  # Play video_z
        current_video = video_z
        video_z.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        video_playing = True
        static_lip_bbox = None  # Force detection of lips
        current_text = "bad"
        text_alpha = 0  # Reset animation
        text_direction = 1
        print("Playing video_z")

cap.release()
video_x.release()
video_z.release()
cv2.destroyAllWindows()
