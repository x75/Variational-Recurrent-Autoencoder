#!/usr/bin/env python

import VRAE
import wave
import numpy as np

from scipy.io import wavfile as wavfile
from sklearn.feature_extraction.image import extract_patches_2d

hidden_units_encoder = 30
hidden_units_decoder = hidden_units_encoder
features = 10
latent_variables = 5
b1 = 1.
b2 = 1.
learning_rate = 0.01
sigma_init = 1e-3
batch_size = 1

patch_size = (64, 1)

vrae = VRAE.VRAE(hidden_units_encoder, hidden_units_decoder, features, latent_variables, b1, b2, learning_rate, sigma_init, batch_size)

(wavrate, wavdata) = wavfile.read("drinksonus44.wav")

wavdata = np.atleast_2d(wavdata).T
# wavdata = wavdata.reshape((1, wavdata.shape[0], wavdata.shape[1]))
print("wavdata.shape", wavdata.shape)

data = extract_patches_2d(wavdata[:,:], patch_size, max_patches=None)
data = data.reshape(data.shape[0], -1)
# print("data.shape", data.shape)


vrae.create_gradientfunctions(data)
