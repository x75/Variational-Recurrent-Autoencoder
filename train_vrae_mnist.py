"""
Authors: 
Joost van Amersfoort - <joost.van.amersfoort@gmail.com>
Otto Fabius - <ottofabius@gmail.com>

#License: MIT
"""

"""This script trains an auto-encoder on the MNIST dataset and keeps track of the lowerbound"""

#python trainmnist.py -s mnist.npy

# import VariationalAutoencoder
import VRAE
import numpy as np
import theano
import argparse
import time
import gzip, cPickle

import matplotlib.pylab as pl

parser = argparse.ArgumentParser()
parser.add_argument("-d","--double", help="Train on hidden layer of previously trained AE - specify params", default = False)

args = parser.parse_args()

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

# data = x_train
data = x_train.T
data = data.reshape((1, data.shape[0], data.shape[1]))

print(data.shape)
print("data[0]", data[0])

dimZ = latent_variables = 20
HU_decoder = 400
HU_encoder = HU_decoder

batch_size = 100
L = 1
learning_rate = 1e-3

if args.double:
    print 'computing hidden layer to train new AE on'
    prev_params = np.load(args.double)
    data = (np.tanh(data.dot(prev_params[0].T) + prev_params[5].T) + 1) /2
    x_test = (np.tanh(x_test.dot(prev_params[0].T) + prev_params[5].T) +1) /2

# [N,dimX] = data.shape
[blub, dimX, N] = data.shape
# encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)
encoder = VRAE.VRAE(HU_decoder,HU_encoder,dimX,dimZ,1., 1., learning_rate,
                    1e-3, batch_size)


if args.double:
    encoder.continuous = True

print "Creating Theano functions"
# encoder.createGradientFunctions()
# data = data.reshape((1, data.shape[0], data.shape[1]))
tdata = theano.shared(data.astype(np.float32))
encoder.create_gradientfunctions(tdata)

# pl.plot(data[0])

print("encoding")
z, mu_encoder, log_sigma_encoder = encoder.encode((data[0].T)[:1000])

print("z.shape, z, mu_enc, s_enc", z.shape, mu_encoder, log_sigma_encoder)
np.save("z.npy", z)

pl.plot(z)
pl.show()

print("decoding")
x = encoder.decode(10, latent_variables, z)
print("x.shape, x", x.shape, x)

pl.plot(x)
pl.show()


# print "Initializing weights and biases"
# encoder.initParams()
# lowerbound = np.array([])
# testlowerbound = np.array([])

# begin = time.time()
# for j in xrange(1500):
#     encoder.lowerbound = 0
#     print 'Iteration:', j
#     encoder.iterate(data)
#     end = time.time()
#     print("Iteration %d, lower bound = %.2f,"
#           " time = %.2fs"
#           % (j, encoder.lowerbound/N, end - begin))
#     begin = end

#     if j % 5 == 0:
#         print "Calculating test lowerbound"
#         testlowerbound = np.append(testlowerbound,encoder.getLowerBound(x_test))
