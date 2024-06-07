# Sequential Characterization of SSVEP EEG signals
This repository contains the relevant code for my master's thesis: "EEG signals for BCIs: sequential characterization and event detection"

The repository is organized as follows:
- Many repetitive and long functions have been implemented in separate python files for easier usage. These files are in the *Python functions* folder
- The actual project's results and methods are presented within jupyter notebooks. These are the notebooks and their corresponding content:
    1. Data_Visualization.ipynb: this notebook contains some of the early plots and trials of dataset handling that we did in order to get a general visualziation of what we were working with.
    2. Initial_Analysis.ipynb: this notebook contains the code for performing our initial analysis done to select the data of interset (PSDA methods, CCA...) and get a better idea of what we would expect by generating synthetic signals. 
    3. Sequential_Anlysis.ipynb: this notebook contains the code we eneded up using to perform the sequential characterization of our selected data.
