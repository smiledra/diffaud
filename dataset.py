from os.path import join
import pickle

from torch.utils import data
import numpy as np
from random import random, randrange, sample
import librosa

class CocktailAudioDataset(data.Dataset):
    def __init__(self, home_dir, clean_dir, noise_dir, sampling=1., snr=1., shuffle_utts=False):
        '''
        home_dir: str; path for AudSpec
        partition: list of tuples; home_dir/partition[i] should contains all .p files
            order: (noise, clean)
        sampling: the portion of entire data to take
        '''
        self.snr = snr
        self.clean_dir = join(home_dir, clean_dir)
        self.noise_dir = join(home_dir, noise_dir)

        with open(self.clean_dir, 'r') as f:
            self.clean_utts = [line.strip('\n') for line in f.readlines()]
        with open(self.noise_dir, 'r') as f:
            self.noise_utts = [line.strip('\n') for line in f.readlines()]
      
        if sampling != 1.: # Subsample
            self.clean_utts = self.clean_utts[:int(len(self.clean_utts)*sampling)]
            self.clean_utt2len = {u: self.clean_utt2len[u] for u in self.clean_utts}

    def __len__(self):
        return len(self.clean_utts)
        
    def __getitem__(self, index_c):
        x_len = librosa.get_duration(path=self.clean_utts[index_c])
        start = random()*(x_len-1)
        xc, _ = librosa.load(path=self.clean_utts[index_c], sr=16000, offset=start, duration=1.0) 
      
        noise_utt = sample(self.noise_utts, 1)[0]
        x_len = librosa.get_duration(path=noise_utt)
        start = random()*(x_len-1)
        xn, _ = librosa.load(path=noise_utt, sr=16000, offset=start, duration=1.0) 
      
        return xn/self.snr+xc, xc

class AudioDataset(data.Dataset):
    def __init__(self, home_dir, partition, sampling=1., shuffle_utts=False):
        '''
        home_dir: str; path for textfiles
        partition: str; filename for the txt of wav files
        sampling: the portion of entire data to take
        '''
        
        self.data_dir = join(home_dir, partition)
        with open(self.data_dir, 'r') as f:
            self.utts = [line.strip('\n') for line in f.readlines()]

        if sampling != 1.: # Subsample
            self.utts = self.utts[:int(len(self.utts)*sampling)]

    def __len__(self):
        return len(self.utts)
        
    def __getitem__(self, index, num_frames=200):
        x_len = librosa.get_duration(path=self.utts[index])
        start = random()*(x_len-1)
        x, _ = librosa.load(path=self.utts[index], sr=16000, offset=start, duration=1.0)
        return x


class AudSpecDataset(data.Dataset):
    def __init__(self, home_dir, partition, sampling=1., shuffle_utts=False):
        '''
        home_dir: str; path for AudSpec
        partition: list of tuples; home_dir/partition[i] should contains all .p files
            order: (noisy, clean)
        sampling: the portion of entire data to take
        '''
        noisy, clean = partition
        self.noisy_dir = join(home_dir, noisy)
        self.clean_dir = join(home_dir, clean)
        with open(join(self.noisy_dir, 'lengths.p'), 'rb') as f:
            self.utt2len = pickle.load(f)
        self.utts = list(self.utt2len.keys())

        if sampling != 1.: # Subsample
            self.utts = self.utts[:int(len(self.utts)*sampling)]
            self.utt2len = {u: self.utt2len[u] for u in self.utts}

    def __len__(self):
        return len(self.utts)
        
    def __getitem__(self, index, num_frames=200):
        with open(join(self.noisy_dir, self.utts[index]), 'rb') as f:
            xn = pickle.load(f)
        with open(join(self.clean_dir, self.utts[index]), 'rb') as f:
            xc = pickle.load(f)
        l = self.utt2len[self.utts[index]]
        start = randrange(0, l-num_frames)
        xn = np.array(xn)[start:start+200, :]
        xc = np.array(xc)[start:start+200, :]
        return xn, xc

class CocktailDataset(data.Dataset):
    def __init__(self, home_dir, partition, sampling=1., snr=1., shuffle_utts=False):
        '''
        home_dir: str; path for AudSpec
        partition: list of tuples; home_dir/partition[i] should contains all .p files
            order: (noise, clean)
        sampling: the portion of entire data to take
        '''
        noise, clean = partition
        self.snr = snr
        self.noise_dir = join(home_dir, noise)
        self.clean_dir = join(home_dir, clean)
      
        with open(join(self.noise_dir, 'lengths.p'), 'rb') as f:
            self.noise_utt2len = pickle.load(f)
        self.noise_utts = list(self.noise_utt2len.keys())
        if sampling != 1.: # Subsample
            self.noise_utts = self.noise_utts[:int(len(self.noise_utts)*sampling)]
            self.noise_utt2len = {u: self.noise_utt2len[u] for u in self.noise_utts}

        with open(join(self.clean_dir, 'lengths.p'), 'rb') as f:
            self.clean_utt2len = pickle.load(f)
        self.clean_utts = list(self.clean_utt2len.keys())
        if sampling != 1.: # Subsample
            self.clean_utts = self.clean_utts[:int(len(self.clean_utts)*sampling)]
            self.clean_utt2len = {u: self.clean_utt2len[u] for u in self.clean_utts}

    def __len__(self):
        return len(self.clean_utts)
        
    def __getitem__(self, index_c, num_frames=200):
        index_n = randrange(0, len(self.noise_utts))
        with open(join(self.noise_dir, self.noise_utts[index_n]), 'rb') as f:
            xn = pickle.load(f)
        with open(join(self.clean_dir, self.clean_utts[index_c]), 'rb') as f:
            xc = pickle.load(f)
        
        start_n = randrange(0, self.noise_utt2len[self.noise_utts[index_n]]-num_frames)
        start_c = randrange(0, self.clean_utt2len[self.clean_utts[index_c]]-num_frames)
      
        xn = np.array(xn)[start_n:start_n+200, :]
        xc = np.array(xc)[start_c:start_c+200, :]
        return xn/self.snr+xc, xc
