import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np 
import torch
from scipy.io.wavfile import read
MATPLOTLIB_FLAG = False

# wandb로 교체 예정
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logger = logging 


from torchaudio.transforms import MelSpectrogram

class MelSpectrogramFixed(torch.nn.Module):  
    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001) 
        return outputs[..., :-1] 


# "sampling_rate": 16000,
# "filter_length": 1280,
# "hop_length": 320,
# "win_length": 1280,
# "n_mel_channels": 80,
# "mel_fmin": 0,
# "mel_fmax": 8000 

# sample_rate=hps.data.sampling_rate,
# n_fft=hps.data.filter_length,
# win_length=hps.data.win_length,
# hop_length=hps.data.hop_length,
# f_min=hps.data.mel_fmin,
# f_max=hps.data.mel_fmax,
# n_mels=hps.data.n_mel_channels,
# window_fn=torch.hann_window





# ============================================================ #

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
