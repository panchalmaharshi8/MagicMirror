import tkinter as tk
import logging
import os

# Configure logging to output to a file
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Ensure the "trial results" directory exists
os.makedirs("trial_results", exist_ok=True)

# Function to handle button clicks
def record_selection(round_num, choice):
    logging.info(f"Round {round_num}: User selected '{choice}'")
    with open(f"trial_results/Trial_{user_name.get()}.txt", "a") as file:
        file.write(f"Round {round_num}: User selected '{choice}'\n")
    if round_num < 5:
        update_screen(round_num + 1)
    else:
        root.quit()  # Exit after round 5

# Function to update the screen text for the current round
def update_screen(round_num):
    if round_num == 1:
        label.config(text="What did you select for round 1?")
        button_left.config(text="Brass", command=lambda: record_selection(1, "Brass"))
        button_right.config(text="Grass", command=lambda: record_selection(1, "Grass"))
    elif round_num == 2:
        label.config(text="What did you select for round 2?")
        button_left.config(text="School", command=lambda: record_selection(2, "School"))
        button_right.config(text="Pool", command=lambda: record_selection(2, "Pool"))
    elif round_num == 3:
        label.config(text="What did you select for round 3?")
        button_left.config(text="Steal", command=lambda: record_selection(3, "Steal"))
        button_right.config(text="Meal", command=lambda: record_selection(3, "Meal"))
    elif round_num == 4:
        label.config(text="What did you select for round 4?")
        button_left.config(text="Bad", command=lambda: record_selection(4, "Bad"))
        button_right.config(text="Dad", command=lambda: record_selection(4, "Dad"))
    elif round_num == 5:
        label.config(text="What did you select for round 5?")
        button_left.config(text="Tin", command=lambda: record_selection(5, "Tin"))
        button_right.config(text="Bin", command=lambda: record_selection(5, "Bin"))

# Function to start the rounds after getting the user's name
def start_rounds():
    user_name.set(entry_name.get())
    with open(f"trial_results/Trial_{user_name.get()}.txt", "w") as file:
        file.write(f"User ID: {user_name.get()}\n")
    entry_name.pack_forget()
    button_start.pack_forget()
    update_screen(1)

# Create the main Tkinter window
root = tk.Tk()
root.title("Selection Prompt")
root.geometry("400x200")

# Create a label for the screen text
label = tk.Label(root, text="Please enter your User ID:", font=("Arial", 14), wraplength=350)
label.pack(pady=20)

# Create an entry widget for the user's name
user_name = tk.StringVar()
entry_name = tk.Entry(root, textvariable=user_name, font=("Arial", 12))
entry_name.pack(pady=10)

# Create a button to start the rounds
button_start = tk.Button(root, text="Start", font=("Arial", 12), command=start_rounds)
button_start.pack(pady=10)

# Create the buttons for the options
button_left = tk.Button(root, text="", font=("Arial", 12))
button_left.pack(side=tk.LEFT, padx=30, pady=20)

button_right = tk.Button(root, text="", font=("Arial", 12))
button_right.pack(side=tk.RIGHT, padx=30, pady=20)

# Start the Tkinter event loop
root.mainloop()
