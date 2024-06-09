import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.signal as signal

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

def plot_generation_spectrogram_wavelet(eeg_data, sampling_rate, freq_idx, stimulus_frequencies, show_plot, save_plot, path, shading_size, shading_sample):

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

    if shading_sample != -1:

        section_start = shading_sample/sampling_rate
        section_end = (shading_sample + shading_size)/sampling_rate

        print(section_start, section_end)
        color = 'lightblue'
        ax1.axvspan(section_start, section_end, facecolor=color, alpha=0.5)




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