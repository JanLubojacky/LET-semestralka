import os

import numpy as np
import pandas as pd

print(pd.__version__)

import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import decimate
from scipy.fft import fft


ORIGINAL_SAMPLE_RATE = 16_000 # 16 kHz

DECIMATE_FACTORS = [8] # [0, 2, 4, 8] # 8, 4, 2 and 1 kHz

datapath = "data/two_commands"

labels = ["up", "down"]

def create_dataset(path, labels):

    for decimate_factor in DECIMATE_FACTORS:

        df = pd.DataFrame()
        
        for i, label in enumerate(labels):

            class_files_path = os.path.join(datapath, label)

            data = []

            for file in os.listdir(class_files_path):

                _, sample = wavfile.read(os.path.join(class_files_path, file))

                # extend to 1s
                sample = np.append(sample, np.array([0] * (ORIGINAL_SAMPLE_RATE - len(sample))))

                decimated_sample = decimate(sample, decimate_factor, n = 4)

                # spectrum plots
                # spectrum = fft(sample)
                # decim_spectrum = fft(decimated_sample)
                
                # plt.figure()
                # plt.subplot(2,2,1)
                # plt.plot(sample)
                # plt.subplot(2,2,3)
                # plt.plot(decimated_sample)
                # plt.subplot(2,2,2)
                # plt.plot(spectrum)
                # plt.subplot(2,2,4)
                # plt.plot(decim_spectrum)
                # plt.show()
                # return 0

                data.append(decimated_sample)
                
            df_temp = pd.DataFrame({"data" : data, "labels" : np.array([i] * len(data))})

            df = pd.concat([df, df_temp], ignore_index = True)

        df.to_pickle(f"speech_two_class_{decimate_factor}.pkl", protocol = 4)


if __name__ == "__main__":
    create_dataset(datapath, labels)
