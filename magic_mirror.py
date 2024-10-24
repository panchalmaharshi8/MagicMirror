import cv2
import mediapipe as mp
import speech_recognition as sr
import threading

# Initialize MediaPipe Face Mesh for both live feed and pre-recorded video
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Dictionary to store trigger phrases and their corresponding video paths
trigger_videos = {
    "he forgot": 'audio_clips/mars/Pizza.mov', #He forgot, it could be amnesia
    "go shopping": "audio_clips/mars/Pineapple.mov", #Go shopping, can you please buy apples?
    "they broke up" : 'audio_clips/mars/Pepperoni.mov', #They broke up, now he pays alimony
    "it's cold": 'audio_clips/mars/Delivery.mov', #It's cold, I'm so shivery
    "i caught a fish": 'audio_clips/mars/Toppings.mov' #I caught a fish, it kept flopping around
}

# Flag and variables for managing state
exit_flag = False
phrase_detected = False
show_overlay = False
current_video = None  # Store the current video object
current_phrase = ""   # Store the detected phrase

# Function to constantly listen for trigger phrases
def detect_phrase():
    global exit_flag, phrase_detected, show_overlay, current_video, current_phrase
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust mic sensitivity
        while not exit_flag:
            try:
                print("Listening for speech...")
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                text = recognizer.recognize_google(audio).lower()
                print("You said:", text)

                # Check if the detected text matches any trigger phrases
                for phrase, video_path in trigger_videos.items():
                    if phrase in text:  # Removed "and not current_video" condition
                        print(f"Trigger detected: '{phrase}'")
                        current_video = cv2.VideoCapture(video_path)  # Load the corresponding video
                        phrase_detected = True
                        show_overlay = True
                        current_phrase = phrase
                        break
            except sr.WaitTimeoutError:
                continue  # Timeout means no phrase detected, continue listening
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

# Start speech recognition in a separate thread
def start_speech_recognition():
    threading.Thread(target=detect_phrase, daemon=True).start()

# Extract lip region from a frame using MediaPipe Face Mesh
def extract_lip_region(frame, face_landmarks, w, h):
    # Extract lip landmarks (indices from MediaPipe)
    outer_lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    lip_region = []

    for idx in outer_lip_idx:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        lip_region.append((x, y))
    
    # Calculate bounding box of the lip region
    x_min = min([p[0] for p in lip_region])
    y_min = min([p[1] for p in lip_region])
    x_max = max([p[0] for p in lip_region])
    y_max = max([p[1] for p in lip_region])

    return (x_min, y_min, x_max, y_max), lip_region

# Overlay lips from pre-recorded video onto the live feed
def overlay_lips(frame, live_lip_bbox, overlay_frame):
    (x_min, y_min, x_max, y_max) = live_lip_bbox

    # Check if the overlay frame is valid (not empty)
    if overlay_frame is None or overlay_frame.size == 0:
        print("Error: Overlay frame is empty or invalid!")
        return  # Skip the overlay process if the frame is invalid
    
    # Extend the box slightly for a better fit
    extension_factor = 2.3  # Adjusted extension factor to make the overlay bigger
    box_width = (x_max - x_min)
    box_height = (y_max - y_min)

    x_min = int(x_min - (box_width * (extension_factor - 1) / 2))
    y_min = int(y_min - (box_height * (extension_factor - 1) / 2))
    x_max = int(x_max + (box_width * (extension_factor - 1) / 2))
    y_max = int(y_max + (box_height * (extension_factor - 1) / 2))

    # Resize the pre-recorded lips to fit the extended bounding box
    resized_overlay = cv2.resize(overlay_frame, (x_max - x_min, y_max - y_min))

    # Extract the lip region from the frame
    lip_area = frame[y_min:y_max, x_min:x_max]

    # Create a mask from the overlay to apply alpha blending
    gray_overlay = cv2.cvtColor(resized_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask to get the non-lip area
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of lips in the original frame
    img_bg = cv2.bitwise_and(lip_area, lip_area, mask=mask_inv)

    # Extract the lip part from the overlay
    img_fg = cv2.bitwise_and(resized_overlay, resized_overlay, mask=mask)

    # Blend the overlay with the original lip area
    blended_lip_area = cv2.add(img_bg, img_fg)

    # Replace the lip region in the original frame with the blended area
    frame[y_min:y_max, x_min:x_max] = blended_lip_area

# Open the camera
cap = cv2.VideoCapture(0)

# Start the speech recognition function
start_speech_recognition()  # Start listening in the background constantly

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    h, w, _ = frame.shape

    # Convert the frame to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for facial landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract lip region from the live feed
            live_lip_bbox, lip_points = extract_lip_region(frame, face_landmarks, w, h)

            if show_overlay and phrase_detected:
                ret_lip, lip_frame = current_video.read()
                if not ret_lip:
                    # If the video ends, reset it or close it and reset flags
                    current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    phrase_detected = False
                    show_overlay = False
                    video_playing = False  # Stop the video and return to listening

                # Overlay the lip video (scaled and extended)
                overlay_lips(frame, live_lip_bbox, lip_frame)
            else:
                # Draw head silhouette (slightly smaller ellipse)
                center_x, center_y = int(w / 2), int(h / 3)
                cv2.ellipse(frame, (center_x, center_y), (100, 150), 0, 0, 360, (0, 255, 0), 2)

            # Draw green dots on the lips
            for (x, y) in lip_points:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Display the mirrored "reflection" feed with lip overlay and/or silhouette
    cv2.imshow("Lip Overlay with Video Trigger and Head Silhouette", frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if current_video is not None:
    current_video.release()
cv2.destroyAllWindows()



# import cv2
# import mediapipe as mp
# import speech_recognition as sr
# import threading

# # Initialize MediaPipe Face Mesh for both live feed and pre-recorded video
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# # Initialize speech recognizer
# recognizer = sr.Recognizer()
# mic = sr.Microphone()

# # Load the pre-recorded video of lips
# lip_video_path = 'audio_clips/Parsa/Pepperoni.mov'  # Replace with your actual video path
# lip_video = None  # This will be initialized when the phrase is detected

# # Flags for managing state
# exit_flag = False
# phrase_detected = False  # To check if the phrase was detected
# show_overlay = False     # To manage when to show overlay or silhouette/dots
# video_playing = False    # To check if the video is currently playing

# # Function to constantly listen for the phrase "the cat"
# def detect_phrase():
#     global exit_flag, phrase_detected, show_overlay, video_playing
#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source)  # Adjust mic sensitivity
#         while not exit_flag:
#             try:
#                 print("Listening for speech...")
#                 audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
#                 text = recognizer.recognize_google(audio).lower()
#                 print("You said:", text)
#                 if "a chef" in text and not video_playing:  # Only trigger if the video isn't already playing
#                     print("Trigger detected: 'a chef'")
#                     phrase_detected = True
#                     show_overlay = True
#                     video_playing = True
#             except sr.WaitTimeoutError:
#                 continue  # Timeout means no phrase detected, continue listening
#             except sr.UnknownValueError:
#                 print("Could not understand audio.")
#             except sr.RequestError as e:
#                 print(f"Could not request results from Google Speech Recognition service; {e}")

# # Start speech recognition in a separate thread
# def start_speech_recognition():
#     threading.Thread(target=detect_phrase, daemon=True).start()

# # Extract lip region from a frame using MediaPipe Face Mesh
# def extract_lip_region(frame, face_landmarks, w, h):
#     # Extract lip landmarks (indices from MediaPipe)
#     outer_lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
#     lip_region = []

#     for idx in outer_lip_idx:
#         x = int(face_landmarks.landmark[idx].x * w)
#         y = int(face_landmarks.landmark[idx].y * h)
#         lip_region.append((x, y))
    
#     # Calculate bounding box of the lip region
#     x_min = min([p[0] for p in lip_region])
#     y_min = min([p[1] for p in lip_region])
#     x_max = max([p[0] for p in lip_region])
#     y_max = max([p[1] for p in lip_region])

#     return (x_min, y_min, x_max, y_max), lip_region

# # Overlay lips from pre-recorded video onto the live feed
# def overlay_lips(frame, live_lip_bbox, overlay_frame):
#     (x_min, y_min, x_max, y_max) = live_lip_bbox

#     # Extend the box slightly for a better fit
#     extension_factor = 2.3  # Adjusted extension factor to make the overlay bigger
#     box_width = (x_max - x_min)
#     box_height = (y_max - y_min)

#     x_min = int(x_min - (box_width * (extension_factor - 1) / 2))
#     y_min = int(y_min - (box_height * (extension_factor - 1) / 2))
#     x_max = int(x_max + (box_width * (extension_factor - 1) / 2))
#     y_max = int(y_max + (box_height * (extension_factor - 1) / 2))

#     # Resize the pre-recorded lips to fit the extended bounding box
#     resized_overlay = cv2.resize(overlay_frame, (x_max - x_min, y_max - y_min))

#     # Extract the lip region from the frame
#     lip_area = frame[y_min:y_max, x_min:x_max]

#     # Create a mask from the overlay to apply alpha blending
#     gray_overlay = cv2.cvtColor(resized_overlay, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)

#     # Invert the mask to get the non-lip area
#     mask_inv = cv2.bitwise_not(mask)

#     # Black-out the area of lips in the original frame
#     img_bg = cv2.bitwise_and(lip_area, lip_area, mask=mask_inv)

#     # Extract the lip part from the overlay
#     img_fg = cv2.bitwise_and(resized_overlay, resized_overlay, mask=mask)

#     # Blend the overlay with the original lip area
#     blended_lip_area = cv2.add(img_bg, img_fg)

#     # Replace the lip region in the original frame with the blended area
#     frame[y_min:y_max, x_min:x_max] = blended_lip_area

# # Draw vertical and horizontal alignment lines
# def draw_alignment_lines(frame, w, h):
#     # Draw a vertical line in the center of the frame
#     center_x = int(w / 2)
#     cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 0), 1)

#     # Draw a horizontal line at 1/8th of the ellipse height from the bottom
#     center_y = int(h / 3)  # Top third of the frame
#     ellipse_height = 150
#     horizontal_y = center_y + int(ellipse_height / 2) - int(ellipse_height / 8)
#     cv2.line(frame, (0, horizontal_y), (w, horizontal_y), (0, 255, 0), 1)

# # Open the camera
# cap = cv2.VideoCapture(0)

# # Start the speech recognition function
# start_speech_recognition()  # Start listening in the background constantly

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     # Flip the frame to create a mirror effect
#     frame = cv2.flip(frame, 1)

#     # Get frame dimensions
#     h, w, _ = frame.shape

#     # Convert the frame to RGB (MediaPipe requires RGB format)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame for facial landmarks
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract lip region from the live feed
#             live_lip_bbox, lip_points = extract_lip_region(frame, face_landmarks, w, h)

#             if show_overlay and phrase_detected:
#                 if lip_video is None:
#                     # Load the lip video once the phrase is detected
#                     lip_video = cv2.VideoCapture(lip_video_path)
#                 ret_lip, lip_frame = lip_video.read()
#                 if not ret_lip:
#                     # If the video ends, reset it or close it and reset flags
#                     lip_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     ret_lip, lip_frame = lip_video.read()
#                     phrase_detected = False
#                     show_overlay = False
#                     video_playing = False  # Stop the video and return to listening

#                 # Overlay the lip video (scaled and extended)
#                 overlay_lips(frame, live_lip_bbox, lip_frame)
#             else:
#                 # Draw head silhouette (slightly smaller ellipse)
#                 center_x, center_y = int(w / 2), int(h / 3)
#                 cv2.ellipse(frame, (center_x, center_y), (100, 150), 0, 0, 360, (0, 255, 0), 2)

#             # Draw green dots on the lips
#             for (x, y) in lip_points:
#                 cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

#     # Draw vertical and horizontal alignment lines
#     draw_alignment_lines(frame, w, h)

#     # Display the mirrored "reflection" feed with lip overlay and/or silhouette
#     cv2.imshow("Lip Overlay with Video Trigger and Head Silhouette", frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# if lip_video is not None:
#     lip_video.release()
# cv2.destroyAllWindows()
