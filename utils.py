import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.signal as signal
import matplotlib.patches as mpatches
import h5py

## -------------------------------------- Data storing and loading functions --------------------------------------------------------------------------

def store_data(file_name, data):
    """
    This function stores the data from an array into an h5 file.
    
    Parameters:
    file_name (str): name of the file the data will be stored into
    data (List[float]): array of data to store (can be multidimensional)
    """
    if file_name.endswith('.h5'):

        with h5py.File(file_name, "w") as file:
            file.create_dataset("data", data=data)

        print("Data stored succesfully in", file_name)

    else:
        print("File name must have extension .h5")


def load_data(file_name):
    """
    This function loads the data from a .h5 file into an array.

    Parameters:
    file_name (str): name of the file the data will be loaded from
    """

    if file_name.endswith('.h5'):

        # Open the HDF5 file in read mode
        with h5py.File(file_name, "r") as file:
            # Load the dataset
            loaded_data = file["data"][:]

        print("Data loaded successfully from", file_name)

        return loaded_data
    
    else:
        print("File name must have extension .h5")
        return []
    



## -------------------------------------- Pandas formatting functions --------------------------------------------------------------------------
def color_numbers_true_false(val):
    if val == 1:
        return 'color: green'
    elif val == 0:
        return 'color: red'
    else:
        return ''

## -------------------------------------- DATASET HANDLING FUNCTIONS --------------------------------------------------------------------------

def load_ssvep_data(dataset_path, subject):
    """
    This function obtains the data from the dataset's mat files given the subject number. It returns the original EEG recordings as well as the EEG recording without resting periods and the trial length
    
    Parameters:
    dataset_path (str): Path to the dataset
    subject (float): Subject number

    Returns:
    Tuple[List[List[List[List[float]]]], List[List[List[List[float]]]], float]:
        Original EEG data (4D array: channel x data x trial x frequency index)
        EEG data without resting periods (4D array: channel x data x trial x frequency index)
        Trial length
    """

    # For S1-S15, the time window is 2 s and the trial length is 3 s, whereas for S16-S70 the time window is 3 s 
    # and the trial length is 4 s.

    file_name = dataset_path

    if dataset_path[-1] != "/":
        file_name += "/"

    trial_length = 3

    file_name += 'S' + str(subject) + '.mat'

    #
    #if subject <= 10:
    #    file_name += 'S1-S10/S' + str(subject) + '.mat'
    #elif subject <= 20:
    #    file_name += 'S11-S20/S' + str(subject) + '.mat'
    #elif subject <= 30:
    #    file_name += 'S21-S30/S' + str(subject) + '.mat'
    #elif subject <= 40:
    #    file_name += 'S31-S40/S' + str(subject) + '.mat'
    #elif subject <= 50:
    #    file_name += 'S41-S50/S' + str(subject) + '.mat'
    #elif subject <= 60:
    #    file_name += 'S51-S60/S' + str(subject) + '.mat'
    #elif subject <= 70:
    #    file_name += 'S61-S70/S' + str(subject) + '.mat'

    if subject > 15:
        trial_length = 4

    data = scipy.io.loadmat(file_name)
    sampling_rate = 250
    time_without_stimuli = 0.5

    # Electrodes data
    eeg_data = data['data'][0, 0]['EEG']

    # Removing time without stimuli:
    # 0.5 seconds without stimuli and 250 sampling rate -> 0.5 * 250 samples without stimuli 
    x = int(time_without_stimuli * sampling_rate)

    # We remove x samples from the beginning and the end. Now we have 500 samples 
    eeg_data_stimulus_only = eeg_data[:, x:750-x, :, :]

    return eeg_data, eeg_data_stimulus_only, trial_length

def load_ssvep_additional_info(path):
    """
    This function returns additional info about the dataset: the corresponding frequency in Hz for each frequency index and the channel names
    
    Parameters:
    path (str): path of one of the dataset's files
    
    Returns:
    Tuple[List[float], List[str]]:
    Frequency in Hz for each frequency index
    Channel names
    """

    # It's the same for all the files
    data = scipy.io.loadmat(path)

    # 40 Frequencies
    data_frequencies = data['data'][0, 0]['suppl_info']['freqs'][0,0][0]

    # All channel names
    ch_names = []
    for i in range(64):
        ch_names.append(data['data'][0, 0]['suppl_info']['chan'][0,0][:, 3][i][0])
        
    ch_names = list(ch_names)

    return data_frequencies, ch_names

def select_occipital_electrodes(electrodes):
    """
    This function selects all electrodes containing an O from the list of channel names (occipital electrodes). It gives this information in three ways.

    Parameters:
    electrodes (List[str]): list of channel names

    Returns: 
    Tuple[List[List[int, str]], List[int], List[str]]:
    List of lists containing electrode index and name for occipital electrodes
    List of electrode indexes for occipital electrodes
    List of names for occipital electrodes
    """

    # We want electrodes with an O
    occipital_electrodes = []
    occipital_indexes = []
    occipital_names = []
    for electrode in range(len(electrodes)):
        name = electrodes[electrode]
        if "O" in name:
            occipital_electrodes.append([electrode, name])
            occipital_indexes.append(electrode)
            occipital_names.append(name)

    return occipital_electrodes, occipital_indexes, occipital_names

## -------------------------------------- CCA SIGNAL GENERATION FUNCTIONS --------------------------------------------------------------------------

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


## -------------------------------------- SEQUENTIAL ANALYSIS FUNCTIONS ----------------------------------------------------------------------

def plot_generation_spectrogram_wavelet(eeg_data, sampling_rate, freq_idx, stimulus_frequencies, show_plot, save_plot, path, shading_samples, shading_size):

    """ 
    This function generates a plot with three panels on on top of each other. The first panel contains the filtered signal, the second panel contains the spectrogram and the third panel contains the visual representation of the wavelet transform coefficients.

    Parameters:
    eeg_data (List[float]): EEG data to use
    sampling_rate (int): sampling rate
    freq_idx (float): the index of the stimulus frequency used for the given EEG data
    stimulus_frequencies (List[float]): list of stimulus frequencies used
    show_plot (boolean): indicates wether or not to show the plot
    save_plot (boolean): indicates whether or not to save the file as a png
    path (str): path of the saved png file
    shading_samples (List[int]): first shaded sample indexes for both our methods. If the shading sample is -1 that method failed and there will be no shaded area for it.
    shading_size (int): number of samples to shade
    """

    wavelet_name = "morl" # Morlet

    # Filtering
    low_freq = 5
    high_freq = 40
    sampling_rate = 250

    b, a = signal.butter(4, [low_freq, high_freq], fs=sampling_rate, btype='band')
    filtered_data = signal.lfilter(b, a, eeg_data) 

    # Rango de frecuencias a analizar. Hay que dividr por sampling_rate para hacer correctamente el cambio a escala
    frequencies = np.arange(1, 40, 0.01) /sampling_rate 
    # Pasamos de frecuencias a escalas para utilizarlas en la transformada
    scales = pywt.frequency2scale(wavelet_name, frequencies) 
    # Transformada Wavelet
    cwtmatr, freqs = pywt.cwt(filtered_data, scales, wavelet_name) 
    # Multiplicamos por samping_rate para recuperar las frecuencias en Hz
    frequencies = frequencies*sampling_rate 

    t = np.arange(0, len(eeg_data)/sampling_rate, 1/sampling_rate)

    # 3 subplots: señal, espectrograma y transformada wavelet
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 6), sharex=True)

    ax1.plot(t, filtered_data, 'k')
    ax1.set_xlim(0, len(eeg_data) / sampling_rate)  # Adjusted x-axis range
    if len(eeg_data) < 750:
        ax1.set_xticks([0.5, 1, 1.5, 2, 2.5])
    else:
        ax1.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax1.set_ylabel("Amplitude V")
    ax1.set_xlabel("Time (s)")

    colors = ["lightblue", "lightyellow"]
    methods = ["m_PSDA", "CCA"]

    # Create proxy artists for legend
    legend_patches = [mpatches.Patch(color=color, label=method) for color, method in zip(colors, methods)]

    # Add legend
    ax1.legend(handles=legend_patches, loc="upper right")

    for index, start in enumerate(shading_samples):

        if start != -1:

            section_start = start/sampling_rate
            section_end = (start + shading_size)/sampling_rate

            color = colors[index]
            ax1.axvspan(section_start, section_end, facecolor=color, alpha=0.75)

            #ax1.text(section_start + 0.1, 14 - 2*index, methods[index], fontsize=8, color='k')


    # Valore utilizados para mantener la escala de colores del espectrograma
    vmin = -10
    vmax = 10

    Pxx, freqs, bins, im = ax2.specgram(filtered_data, NFFT=300, Fs=sampling_rate, noverlap=int(300/4), cmap='jet', vmin=vmin, vmax=vmax)
    f_value = stimulus_frequencies[freq_idx]
    ticks = np.arange(0, 50, step=f_value)
    ax2.set_yticks(ticks)
    if len(eeg_data) < 750:
        ax2.set_xticks([0.5, 1, 1.5, 2, 2.5])
    else:
        ax2.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_ylim(0, 50)
    # Add colorbar for the spectrogram
    cbar_ax2 = fig.add_axes([1.01, 0.45, 0.02, 0.2])  # [left, bottom, width, height]
    plt.colorbar(im, cax=cbar_ax2, label='Power (dB/Hz)')

    # Valores utilizados para mantener la escala de colores de la transformada wavelet
    vmin = 0 
    vmax = 40

    ax3.imshow(np.abs(cwtmatr), aspect='auto', cmap='jet', extent=[0, len(filtered_data) / sampling_rate, frequencies[-1], frequencies[0]], origin='upper', vmin=vmin, vmax=vmax)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_ylim(2, 35)
    ticks = np.arange(0, 35, step=f_value)
    ax3.set_yticks(ticks)
    ax3.set_xlim(0, len(eeg_data) / sampling_rate)  # Adjusted x-axis range
    cbar_ax3 = fig.add_axes([1.01, 0.11, 0.02, 0.2])
    plt.colorbar(ax3.images[0], cax=cbar_ax3, label="Magnitude")

    plt.tight_layout()
    
    if save_plot:
        plt.savefig(path, dpi=fig.dpi, bbox_inches='tight')

    if show_plot:
        plt.show()
    else: 
        plt.close()

    


def plot_generation_fft(eeg_data, sampling_rate, frequency, show_plot, save_plot, path):

    """
    This function generates a plot with the power spectra of the filtered signal corresponding to the EEG data given. The plot marks the highest peak.
    
    Parameters:
    eeg_data (List[float]): EEG data to use
    sampling_rate (int): sampling rate of the eeg data
    frequency (float): real stimulus frequency
    show_plot (boolean): indicates wether or not to show the plot
    save_plot (boolean): indicates whether or not to save the file as a png
    path (str): path of the saved png file
    """
    # Filtering
    low_freq = 5
    high_freq = 40
    sampling_rate = 250

    b, a = signal.butter(4, [low_freq, high_freq], fs=sampling_rate, btype='band')
    filtered_data = signal.lfilter(b, a, eeg_data)

    fft_result = np.fft.fft(filtered_data)
    freq = np.fft.fftfreq(len(filtered_data), d=1/sampling_rate)
    power_spectra = np.abs(fft_result) ** 2


    plt.plot(freq, power_spectra)
    plt.xlim(0, 400)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power V²")
    plt.title("Power Spectra (" +  str(frequency) +" Hz)")
    plt.plot(freq[np.argmax(power_spectra[0:500])], max(power_spectra[0:500]), marker="o", color="red")
    plt.text(freq[np.argmax(power_spectra[0:500])] + 2, max(power_spectra[0:500]), str(freq[np.argmax(power_spectra[0:500])]) + " Hz")
    plt.xlim(0, 55)
    plt.ylim(0)

    if save_plot:
        plt.savefig(path, dpi='figure', format=None)
    
    if show_plot:
        plt.show()
    else:
        plt.close()

        

def highest_magntiudes(data, frequency, sampling_rate, path):
    
    """
    This function generates a plot showing which frequencies had the highest wavelet transform magnitudes over time filtering out the magnitudes below 25 for easier visual interpretation. The stimulus frequency is marked with a red line.
    Parameters:
    data (List[float]): EEG recording under study
    frequency (float): the stimulus frequency 
    sampling_rate (int): sampling rate
    path (string): path to save the generated plot
    """

    low_freq = 5
    high_freq = 40
    sampling_rate = 250

    b, a = signal.butter(4, [low_freq, high_freq], fs=sampling_rate, btype='band')
    filtered_data = signal.lfilter(b, a, data) 


    freqs = np.arange(1, 40, 0.01) /sampling_rate 
    wavelet_name = "morl"
    # Pasamos de frecuencias a escalas para utilizarlas en la transformada
    scales = pywt.frequency2scale(wavelet_name, freqs) 
    # Transformada Wavelet
    cwtmatr, freqs = pywt.cwt(filtered_data, scales, wavelet_name)

    # Filter out magnitudes below 20
    cwtmatr_filtered = np.where(cwtmatr < 25, 0, cwtmatr)

    # Find the index of the maximum magnitude along the frequency axis for each time step
    max_freq_indices = np.argmax(cwtmatr_filtered, axis=0)

    # Extract the corresponding frequencies for each time step
    frequencies = np.arange(1, 40, step=0.01)
    max_freq_values = frequencies[max_freq_indices]

    t = np.arange(0, cwtmatr.shape[1]/250, 1/250)

    # Plot the time vs. the frequency with the highest magnitude
    plt.plot(t, max_freq_values, label='Frequency with Highest Magnitude (>25)')
    plt.axhline(frequency, color="red", label="Stimulus frequency")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Highest magnitudes over time')
    plt.legend()
    plt.yticks(np.arange(1, 30, step=2))
    if cwtmatr.shape[1] > 800:
        plt.xlim(0, 4)
    else:
        plt.xlim(0, 3)

    if path != "":
        plt.savefig(path, dpi='figure', format=None)
        
    else:    
        plt.show()
    
    plt.close()