import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def smooth_by_conv(window_size, df, col):
    padded_track = pd.concat([pd.DataFrame([[df.iloc[0][col]]]*(window_size//2), columns=[0]),
                     df[col],
                     pd.DataFrame([[df.iloc[-1][col]]]*(window_size//2), columns=[0])])
    smoothed_signals = np.convolve(padded_track.squeeze(), np.ones(window_size)/window_size, mode='valid')
    return smoothed_signals
