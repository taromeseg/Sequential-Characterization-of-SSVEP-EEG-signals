import numpy as np
import scipy.linalg
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

def plot_spectrogram(data, filtered_data, sampling_rate):

    # Calculate the spectrogram with improved frequency resolution for the original data
    frequencies, times, Sxx_data = signal.spectrogram(
        data, fs=sampling_rate, nperseg=int(sampling_rate * 1), window='hann')

    # Calculate the spectrogram with improved frequency resolution for the filtered data
    frequencies, times, Sxx_filtered = signal.spectrogram(
        filtered_data, fs=sampling_rate, nperseg=int(sampling_rate * 1), window='hann')

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the spectrogram of the original data
    cmap_data = plt.get_cmap('Blues')
    img1 = ax1.pcolormesh(times, frequencies, 10 * np.log10(Sxx_data), cmap=cmap_data)
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Original Data Spectrogram')
    ax1.set_ylim(0, 40)

    # Plot the spectrogram of the filtered data
    cmap_filtered = plt.get_cmap('Blues')  # You can choose a different colormap here
    img2 = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx_filtered), cmap=cmap_filtered)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Filtered Data Spectrogram')
    ax2.set_ylim(0, 40)

    # Add colorbars
    fig.colorbar(img1, ax=ax1, label='Power/Frequency (dB/Hz)')
    fig.colorbar(img2, ax=ax2, label='Power/Frequency (dB/Hz)')

    plt.tight_layout()
    plt.show()
        
def save_signal_to_file(eeg_signal=[], file_name="signal.txt"):
    """
    Save an eeg signal into a text file

    Parameters:
    eeg_signal (List[float]): List with the float samples to save
    file_name (string): Name of the file. If it already exists it will be overwritten. Otherwise a new file will be created.
    """

    with open(file_name, 'w') as file:
        for sample in eeg_signal:
            file.write(str(sample) + '\n')


def load_signal_from_file(file_name="signal.txt"):
    """
    Load the sample of a signal from a text file.

    Parameters:
    file_name (int): Name of the file to load the data from. If it doesn't exist an error will happen (FileNotFoundError).

    Returns:
    List[float]: List containing the loaded float samples
    """

    signal = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                sample = float(line.strip())
                signal.append(sample)
    except FileNotFoundError:
        pass

    return signal

# The following functions are used for CCA analysis

PI = np.pi
sin = lambda f, h, t, p: np.sin(2*PI*f*h*t + p) # Function for generating the sin element of the waves
cos = lambda f, h, t, p: np.cos(2*PI*f*h*t + p) # Function for generating the cos element of the waves
reference_signal = lambda f, h, t, p: [sin(f, h, t, p), cos(f, h, t, p)] # Function for generating a reference wave composed of a sin and cos

def generate_ref_signal_at_time(f, t, num_harmonics, phase):
    """
    Generates a time point of a reference signal composed of sines and cosines with the specified number of harmonics.

    Parameters:
    f (float): oscillation frequency of the generated signal
    t (float): time point to generate
    num_harmonics (int): number of harmonics to include
    phase (float): phase to add when generating the signal

    Returns:
    List[float]: generated time point containing both the sine and cosine components with the corresponding harmonics.
    """
    ref_values = []
    for h in range(1, num_harmonics + 1): # Starting with 1. Using harmonic 0 doesn't do anything
        ref_values += reference_signal(f, h, t, phase)
    return ref_values

def generate_ref_signal(f, sampling_rate, duration, num_harmonics, phase):
    """
    Generate a reference signal oscillating at frequency f with the specified sampling rate, duartion, number of harmonics considered and starting phase.

    Parameters:
    f (float): oscillation frequency of the generated signal
    sampling_rate (int): sampling rate of the generated signal
    duration (float): length in seconds of the generated signal
    num_harmonics (int): numer of harmonics to include in the signal 
    phase (float): initial phase to add

    Returns:
    List[float]: generated signal with oscillations at frequency f and the specified harmonics.
    """
    ref_signal = []
    num_samples = duration * sampling_rate
    for sample in range(int(num_samples)):
        t = sample * 1/sampling_rate
        ref_signal.append(generate_ref_signal_at_time(f, t, num_harmonics, phase))
    return ref_signal

# Solve for Maximum CCA from two multidimensional signal
def find_maximum_canonical_correlations(X, Y):
    if len(X) == len(Y):
        N = len(X)
    else:
        print('Unequal shapes: X = ', len(X), ' Y =', len(Y))
        return None
    
    C_xx = 1/N * (X.T @ X)
    C_yy = 1/N * (Y.T @ Y)
    C_xy = 1/N * (X.T @ Y)
    C_yx = 1/N * (Y.T @ X)
    C_xx_inv = np.linalg.pinv(C_xx)
    C_yy_inv = np.linalg.pinv(C_yy)
    eig_values, eig_vectors = scipy.linalg.eig(C_yy_inv @ C_yx @ C_xx_inv @ C_xy)
    sqrt_eig_values = np.sqrt(eig_values)
    return max(sqrt_eig_values)

def perform_CCA(eeg_data, reference_signals, frequencies):

    result = []
    
    total_count = 0

    for i in range(len(frequencies)):

        cca_input = pd.DataFrame(eeg_data[:, i])
        max_cca = {}

        for ref_signal in reference_signals.keys():
            value = find_maximum_canonical_correlations(cca_input, reference_signals[ref_signal])
            max_cca[ref_signal] = value # We input the maximum cca for each of the reference frequencies

        max_cca["Result"] = max(max_cca.items(), key = lambda x: x[1])[0] # We add the detected frequency
        max_cca["Real frequency"] = frequencies[i] # We add the expected frequency

        if (max_cca["Result"] == max_cca["Real frequency"]):
            total_count += 1

        result.append(max_cca)

    accuracy = total_count/len(frequencies)

    #print("Accuracy: ", accuracy)
    table = [["Result"],["Real frequency"]]
    for i in range(len(frequencies)):
        table[0].append(result[i]["Result"])
        table[1].append(result[i]["Real frequency"])
    
    return table, accuracy, max_cca


def perform_sequential_CCA(eeg_data, reference_signals, frequencies, window_size, overlap):
    result = []
    total_count = 0

    for i in range(0, len(frequencies), overlap):
        window_eeg = eeg_data[i:i+window_size]
        window_result = perform_CCA(window_eeg, reference_signals, frequencies)
        result.extend(window_result)

    return result