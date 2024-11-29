# import tkinter as tk
# import logging

# # Configure logging to output to system log
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# # Function to handle button clicks
# def record_selection(question_num, choice):
#     if question_num == "color":
#         logging.info(f"Question: Letter color selected - '{choice}'")
#         update_screen(1)
#     elif question_num <= 3:
#         logging.info(f"Round {question_num}: User selected '{choice}'")
#         if question_num < 3:
#             update_screen(question_num + 1)
#         else:
#             root.quit()  # Exit after round 3

# # Function to update the screen text dynamically
# def update_screen(round_num):
#     if round_num == 1:
#         label.config(text="What did you select for round 1?")
#         button_left.config(text="Dad", command=lambda: record_selection(1, "Dad"))
#         button_right.config(text="Bad", command=lambda: record_selection(1, "Bad"))
#     elif round_num == 2:
#         label.config(text="What did you select for round 2?")
#         button_left.config(text="Dad", command=lambda: record_selection(2, "Dad"))
#         button_right.config(text="Bad", command=lambda: record_selection(2, "Bad"))
#     elif round_num == 3:
#         label.config(text="What did you select for round 3?")
#         button_left.config(text="Dad", command=lambda: record_selection(3, "Dad"))
#         button_right.config(text="Bad", command=lambda: record_selection(3, "Bad"))

# # Create the main Tkinter window
# root = tk.Tk()
# root.title("Selection Prompt")
# root.geometry("400x200")

# # Create a label for the screen text
# label = tk.Label(root, text="What color were the letters on the screen?", font=("Arial", 14), wraplength=350)
# label.pack(pady=20)

# # Create the buttons for the first question
# button_left = tk.Button(root, text="Black", font=("Arial", 12), command=lambda: record_selection("color", "Black"))
# button_left.pack(side=tk.LEFT, padx=30, pady=20)

# button_right = tk.Button(root, text="Blue", font=("Arial", 12), command=lambda: record_selection("color", "Blue"))
# button_right.pack(side=tk.RIGHT, padx=30, pady=20)

# # Start the Tkinter event loop
# root.mainloop()


import tkinter as tk
import logging

# Configure logging to output to system log
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Function to handle button clicks
def record_selection(round_num, choice):
    logging.info(f"Round {round_num}: User selected '{choice}'")
    if round_num < 3:
        update_screen(round_num + 1)
    else:
        root.quit()  # Exit after round 3

# Function to update the screen text for the current round
def update_screen(round_num):
    label.config(text=f"What did you select for round {round_num}?")
    button_dad.config(command=lambda: record_selection(round_num, "Dad"))
    button_bad.config(command=lambda: record_selection(round_num, "Bad"))

# Create the main Tkinter window
root = tk.Tk()
root.title("Selection Prompt")
root.geometry("300x200")

# Create a label for the screen text
label = tk.Label(root, text="What did you select for round 1?", font=("Arial", 14), wraplength=250)
label.pack(pady=20)

# Create the buttons for "Dad" and "Bad"
button_dad = tk.Button(root, text="Dad", font=("Arial", 12), command=lambda: record_selection(1, "Dad"))
button_dad.pack(side=tk.LEFT, padx=30, pady=20)

button_bad = tk.Button(root, text="Bad", font=("Arial", 12), command=lambda: record_selection(1, "Bad"))
button_bad.pack(side=tk.RIGHT, padx=30, pady=20)

# Start the Tkinter event loop
root.mainloop()
