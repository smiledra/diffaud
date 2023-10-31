import argparse
import librosa
from tqdm import tqdm
import pickle
import glob
import time
from functools import partial


import numpy as np
from math import floor, ceil
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap, config, random
config.update("jax_enable_x64", True)

import optax
import flax.linen as nn
from random import Random

from strfpy import *
from strfpy_jax import *
from model import *

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
print(jax.default_backend())

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr_v', required=True, type=float)
  parser.add_argument('--lr_sr', required=True, type=float)
  parser.add_argument('--minibatch_size', default=4, type=int)
  parser.add_argument('--num_frames', default=200, type=int)
  parser.add_argument('--num_epochs', default=200, type=int)
  parser.add_argument('--num_strfs', default=40, type=int)

  parser.add_argument('--num_layers', default=4, type=int)
  parser.add_argument('--num_features', default=[40,20,10,1], nargs="+", type=int)

  parser.add_argument('--apply_attention', default=False, type=bool)
  parser.add_argument('--num_heads', default=None, type=int)

  parser.add_argument('--train_clean_dir', required=True, type=str)
  parser.add_argument('--train_noisy_dir', required=True, type=str)
  parser.add_argument('--save_dir', required=True, type=str)

  config = parser.parse_args()

  @jit
  def forward_loss_batch(variables, v, sr, v_out):
    '''shape of v should be batch x 200 x 128'''
    rf = jnp.abs(strf_batch(v, sr, signs)).transpose(0,3,2,1)
    #print(rf.shape)
    v_hat = model.apply(variables, rf).squeeze(axis=3).transpose(0,2,1)
    return jnp.mean(optax.l2_loss(v_out, v_hat))
      
  def update_batch_sr(variables, sr, ov, osr, opt_state_v, opt_state_sr, v, v_out): 
    loss, (g_v, g_sr) = value_and_grad(forward_loss_batch, (0,2))(variables, v, sr, v_out)
    u_v, opt_state_v = ov.update(g_v, opt_state_v)
    u_sr, opt_state_sr = osr.update(g_sr, opt_state_sr)
    new_sr = optax.apply_updates(sr, u_sr)
    new_sr = jnp.vstack([new_sr[:,0].clip(0.25, 8), new_sr[:,1].clip(2, 32)]).T
    return optax.apply_updates(variables, u_v), new_sr, opt_state_v, opt_state_sr, loss
  
  model = BatchConvSTRF2spec(num_layers=config.num_layers, 
                             num_strides=[(1,1) for _ in range(config.num_layers)], 
                             num_features=config.num_features,
                             #apply_attention=config.apply_attention, 
                             num_heads=config.num_heads
                            )
  
  sr, signs = initialize_strfs(config.num_strfs)
  variables = model.init(jax.random.key(0), 
                         jnp.ones([config.minibatch_size, 128, config.num_frames, len(sr)]))

  train_set_noisy = glob.glob(config.train_noisy_dir)[:100]
  v_noisy = preprocess_train_set(train_set_noisy, sample_stride=50, num_frames=200)
  train_set_clean = glob.glob(config.train_clean_dir)[:100]
  v_clean = preprocess_train_set(train_set_clean, sample_stride=50, num_frames=200)
  
  op_v, op_sr = optax.adam(config.lr_v), optax.adam(config.lr_sr)
  opt_state_v = op_v.init(variables)
  opt_state_sr = op_sr.init(sr)

  print(f'Start training. Shape of training data: {v_clean.shape}')
  for epoch in range(config.num_epochs):
      t0 = time.time()
      total_loss = 0
      for minibatch in range(floor(len(v_clean)/config.minibatch_size)):
          v_minibatch = v_noisy[minibatch:minibatch+config.minibatch_size, :, :]
          v_out = v_clean[minibatch:minibatch+config.minibatch_size,:,:]
          variables, sr, opt_state_v, opt_state_sr, l = update_batch_sr(
              variables, sr, op_v, op_sr, opt_state_v, opt_state_sr, v_minibatch, v_out)
          total_loss += l
      print(f"Epoch: {epoch}; loss: {total_loss*1000}; Time: {(time.time()-t0)/60} min")
  with open(config.save_dir, 'wb') as temp:
      pickle.dump([variables, sr], temp)

def initialize_strfs(n_strfs, proportion_negative_signs=0.5):
  n_strfs = 40
  # Between 0 and 1
  np.random.seed(0)
  sv = np.random.rand(n_strfs, 2)
  # s = np.random.rand(n_strfs)*5-2
  # v = np.random.rand(n_strfs)*4+1
  # sv = np.vstack([jnp.expand_dims(s, axis=0),jnp.expand_dims(v, axis=0)])
  # sv = jnp.array(np.exp2(sv).T)
  
  
  signs = np.ones(n_strfs)
  signs[:floor(n_strfs*proportion_negative_signs)] -= 2
  signs = tuple(np.array(signs))
  plt.scatter(sv[:,0], sv[:,1])
  plt.show()
  return (sv, signs)

def preprocess_train_set(train_set, sample_stride, num_frames, onset_cutoff=100, noisy=False):
  vs = []
  for fname in train_set:
      with open(fname, 'rb') as temp:
          v = pickle.load(temp)[onset_cutoff:, :]
      num_batch = (len(v)-num_frames)//sample_stride
      for minibatch in range(num_batch):
          vs.append(v[sample_stride*minibatch:sample_stride*minibatch+num_frames,:]/7)
  Random(0).shuffle(vs)
  vs = jnp.vstack([jnp.expand_dims(v, axis=0) for v in vs])
  return vs

class ConvSTRF2spec(nn.Module):
  """Convolutional Decoder."""
  num_layers: int
  num_strides: int
  num_features: int
  num_heads: int

  def __call__(self, x):
    return self.conv(x)    
  
  @nn.compact
  def conv(self, x):
    '''Input: frequency (128) x time (200/s) x n_channel (e.g. 198)'''
    for d in range(self.num_layers):
      x = nn.ConvTranspose(features=self.num_features[d], kernel_size=(3, 3), strides=self.num_strides[d])(x) 
      x = nn.gelu(x)
    if self.num_heads!=0:
      x = nn.SelfAttention(
        num_heads=self.num_heads,
        qkv_features=self.num_heads*32,
        use_bias=False
        )(x)
    return x


@partial(jit, static_argnums=2)
def strf(y, sr, signs):
  '''
  Python version of aud2cor() in the NSL toolbox. From auditory spectrogram to cortical STRF.

  Input:
  y: auditory spectrogram
  paras: parameters that generated the auditory spectrogram, e.g. [5, 8, -2, 0]
  rv: rate vector
  sv: scale vector
  out_filename: write the output in this file
  disp: normalization during display
  '''

  sr = sr.at[:,1].apply(lambda x: nn.sigmoid(x)*20+2)
  sr = sr.at[:,0].apply(lambda x: nn.sigmoid(x)*7.75+0.25)

  paras = [5, 8, -2, 0]
  FULLX, FULLT, BP = 0, 0, 0
  STF, SRF = 1000/paras[0], 24
  
  N, M = y.shape
  N1, M1 = 2**ceil(np.log2(N)), 2**ceil(np.log2(M))
  N2, M2 = N1*2, M1*2
  
  Y = jnp.fft.fft(y, M2, axis=1)[:,:M1] # Fourier transform (frequency)
  Y = jnp.fft.fft(Y[:N,:], N2, axis=0) # Fourier transform (temporal)

  cr = []
  for i in range(len(sr)): # rate filtering
    (s, r), sgn = sr[i,:], signs[i]
    HR = gen_cort_strf(r, N1, STF) 

    HR = jnp.concatenate([HR, np.zeros(N1)])
    if sgn == -1: # conjugate
      HR = jnp.insert(jnp.conj(jnp.flipud(HR[1:N2])), 0, HR[0])
      HR = HR.at[N1].set(jnp.abs(HR[N1+1]))

    #print(sgn, HR.shape, Y.T.shape)
    z1 = (HR * Y.T).T # Temporal convolution
    z1 = jnp.fft.ifft(z1, axis=0)[:N, :]

    HS = gen_corf_strf(s, M1, SRF) 
    z1 = z1*HS # Frequency convolution
    
    R1v = jnp.fft.ifft(z1, M2) # Second inverse FFT
    cr.append(jnp.expand_dims(R1v[:, :M], axis=0))
  cr = jnp.vstack(cr)
  return cr

strf_batch = vmap(strf, in_axes=(0, None, None))

BatchConvSTRF2spec = nn.vmap(
    ConvSTRF2spec,
    in_axes=0, out_axes=0,
    variable_axes={'params': 0},
    split_rngs={'params': True},
    methods=["__call__", "conv"])

if __name__ == '__main__':
  main() 
