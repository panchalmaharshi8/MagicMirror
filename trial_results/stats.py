import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Hard-coded data for all trials
data = {
    "Trial 1": {"C": 6, "I": 3},
    "Trial 2": {"C": 8, "I": 1},
    "Trial 3": {"C": 8, "I": 1},
    "Trial 4*": {"C": 7, "I": 2},
    "Trial 5": {"C": 7, "I": 2}
}

# Create contingency table for all trials
contingency_table = np.array([
    [data["Trial 1"]["C"], data["Trial 1"]["I"]],
    [data["Trial 2"]["C"], data["Trial 2"]["I"]],
    [data["Trial 3"]["C"], data["Trial 3"]["I"]],
    [data["Trial 4*"]["C"], data["Trial 4*"]["I"]],
    [data["Trial 5"]["C"], data["Trial 5"]["I"]]
])

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Function to generate the two plots
def generate_chi_square_plots(contingency_table, expected, trials):
    """
    Generate two plots:
    1. Observed frequencies across trials.
    2. Observed vs Expected frequencies, where Expected is one bar.
    """
    # Plot 1: Observed Frequencies Only
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(trials))  # Trial indices for plotting
    width = 0.4  # Bar width

    # Observed bar stacks
    ax1.bar(x, contingency_table[:, 0], width, label='Observed Correct (C)')
    ax1.bar(x, contingency_table[:, 1], width, bottom=contingency_table[:, 0], label='Observed Incorrect (I)')

    # Labeling for observed frequencies
    ax1.set_xlabel('Trials')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Observed Frequencies Across Trials')
    ax1.set_xticks(x)
    ax1.set_xticklabels(trials)

    # Legend off to the side
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust spacing to prevent legend cutoff
    fig1.subplots_adjust(right=0.75)

    # Plot 2: Observed vs Expected Frequencies (Expected as One Bar)
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(trials))  # Trial indices for plotting
    width = 0.4  # Bar width

    # Observed frequencies bar stack
    ax2.bar(x, contingency_table[:, 0], width, label='Observed Correct (C)')
    ax2.bar(x, contingency_table[:, 1], width, bottom=contingency_table[:, 0], label='Observed Incorrect (I)')

    # Expected frequencies as a single bar stack
    ax2.bar(len(trials), expected[:, 0].mean(), width, label='Expected Correct (C)', alpha=0.7, hatch='/')
    ax2.bar(len(trials), expected[:, 1].mean(), width, bottom=expected[:, 0].mean(), label='Expected Incorrect (I)', alpha=0.7, hatch='/')

    # Labeling for observed vs expected
    ax2.set_xlabel('Trials')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Observed vs Expected Frequencies Across Trials')
    ax2.set_xticks(np.append(x, len(trials)))
    ax2.set_xticklabels(trials + ["Expected"])

    # Legend off to the side
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust spacing to prevent legend cutoff
    fig2.subplots_adjust(right=0.8)

    plt.show()

# List of trials
trials = list(data.keys())

# Call the function to generate plots
generate_chi_square_plots(contingency_table, expected, trials)

# Output chi-square test results
print("Chi-Square Test for Homogeneity Across All Trials:\n")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies: \n{expected}\n")
