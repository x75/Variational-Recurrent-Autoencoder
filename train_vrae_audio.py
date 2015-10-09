#!/usr/bin/env python

import VRAE
import wave
import numpy as np
import pylab as pl
import theano

from scipy.io import wavfile as wavfile
from sklearn.feature_extraction.image import extract_patches_2d

hidden_units_encoder = 500
hidden_units_decoder = hidden_units_encoder
features = 1
latent_variables = 100
b1 = 0.05
b2 = 0.001
learning_rate = 1e-3
sigma_init = 1e-3
batch_size = 1


vrae = VRAE.VRAE(hidden_units_encoder, hidden_units_decoder, features, latent_variables, b1, b2, learning_rate, sigma_init, batch_size)

(wavrate, wavdata) = wavfile.read("drinksonus44.wav")

wavdata = np.atleast_2d(wavdata[:10000])
# wavdata = wavdata.reshape((1, wavdata.shape[0], wavdata.shape[1]))
print("wavdata.shape", wavdata.shape)

# patch_size = (64, 1)
# data = extract_patches_2d(wavdata[:,:], patch_size, max_patches=None)
# data = data.reshape(data.shape[0], -1)
wavlen = wavdata.shape[1]
chunklen = 1
numchunks = wavlen/chunklen
data = wavdata[0,:numchunks*chunklen].reshape((1, chunklen, numchunks))
# data = wavdata[0,:numchunks*chunklen].reshape((chunklen, numchunks))
print("data.shape", data.shape)


print("create_gradientfunctions")
data = data.astype(np.float64)
tdata = theano.shared(data)
vrae.create_gradientfunctions(tdata)

print("save_parameters")
vrae.save_parameters("data/")

print("encoding")
# z, mu_encoder, log_sigma_encoder = vrae.encode(data[0,:1].T)
z, mu_encoder, log_sigma_encoder = vrae.encode((data[0].T)[:1000])

print("z.shape, z, mu_enc, s_enc", z.shape, mu_encoder, log_sigma_encoder)
np.save("z.npy", z)

pl.plot(z)
pl.show()

print("decoding")
x = vrae.decode(1000, latent_variables, z)
print("x.shape, x", x.shape, x)

pl.plot(x)
pl.show()

wavfile.write("x.wav", 44100, x)
