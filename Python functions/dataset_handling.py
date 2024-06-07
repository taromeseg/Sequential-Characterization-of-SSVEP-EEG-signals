import scipy.io

def color_numbers_true_false(val):
    if val == 1:
        return 'color: green'
    elif val == 0:
        return 'color: red'
    else:
        return ''

def load_ssvep_data(subject):

    # For S1-S15, the time window is 2 s and the trial length is 3 s, whereas for S16-S70 the time window is 3 s 
    # and the trial length is 4 s.

    file_name = '../Dataset BETA/'

    trial_length = 3

    if subject <= 10:
        file_name += 'S1-S10/S' + str(subject) + '.mat'
    elif subject <= 20:
        file_name += 'S11-S20/S' + str(subject) + '.mat'
    elif subject <= 30:
        file_name += 'S21-S30/S' + str(subject) + '.mat'
    elif subject <= 40:
        file_name += 'S31-S40/S' + str(subject) + '.mat'
    elif subject <= 50:
        file_name += 'S41-S50/S' + str(subject) + '.mat'
    elif subject <= 60:
        file_name += 'S51-S60/S' + str(subject) + '.mat'
    elif subject <= 70:
        file_name += 'S61-S70/S' + str(subject) + '.mat'

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

def load_ssvep_additional_info():

    # It's the same for all the files
    data = scipy.io.loadmat('../Dataset BETA/S1-S10/S1.mat')

    # 40 Frequencies
    data_frequencies = data['data'][0, 0]['suppl_info']['freqs'][0,0][0]

    # All channel names
    ch_names = []
    for i in range(64):
        ch_names.append(data['data'][0, 0]['suppl_info']['chan'][0,0][:, 3][i][0])
        
    ch_names = list(ch_names)

    return data_frequencies, ch_names

def select_occipital_electrodes(electrodes):
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
