import argparse
import librosa
from tqdm import tqdm
import pickle
import glob
import time

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.io.wavfile import write as wavfile_write

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax.scipy.signal import convolve as jconvolve 
from jax import config
config.update("jax_enable_x64", True)

from math import ceil
from scipy.signal import lfilter
from scipy.fft import fft, ifft, fftfreq
from strfpy import *
from strfpy_jax import *
print(f"Running on {jax.default_backend()}...")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--spec_path', required=True, nargs="+", type=str)
  parser.add_argument('--print_interval', default=50, type=int)
  config = parser.parse_args()
  
  Bs, As = read_cochba_j()
  frmlen = 5 # paras(1)
  time_constant = 8 # paras(2)
  fac = -2 # paras(3)
  octave_shift = 0 # paras(4)

  @jit
  def forward_loss(x_hat, v):
    v_hat = wav2aud_j(x_hat, frmlen, time_constant, fac, octave_shift, As, Bs, fft=True, return_stage=5)
    return jnp.sum((v-v_hat)**2)
  
  def update(x_hat, v): 
    l, grad = value_and_grad(forward_loss)(x_hat, v)
    return x_hat - lr * grad, l

  for spec_file in config.spec_path:
    with open(spec_file, 'rb') as f:
      v = pickle.load(f)
  
    duration = 1.0
    len_x = int(16000*duration)
    num_epochs = 300
    lr = 0.025
    print(f"The shape of the auditory spectrogram is {v.shape}")
    
    t0 = time.time()
    x_hat = jnp.array(np.zeros(len_x))
    
    for epoch in range(num_epochs):
      t = time.time()
      x_hat, loss = update(x_hat, v)
      if epoch % config.print_interval == 0:
        print(f"Slice {slice}, epoch {epoch}, loss = {loss}, time passed = {time.time()-t}.")
      if np.abs(loss) < 0.1:
        break
      if epoch % 100 == 0: 
        lr /= 2
    
    print(f"Total time passed: {time.time() - t0}")
    wavfile_write(spec_file[:-1]+'wav', 16000, np.array(x_hat))

if __name__ == '__main__':
  main() 
