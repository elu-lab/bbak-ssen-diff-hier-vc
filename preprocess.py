import os
# import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

import torchaudio
from torchaudio.transforms import MelSpectrogram

import pyworld as pw

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic 

from utils.utils import *

# ============================================================ #

def load_audio_to_torch(audio_path, is_inference=False):
    # torchaudio load
    wav, sr = torchaudio.load(audio_path, 
                            #   normalize=True (Default)
                            )

    # resample
    sample_rate = 16000
    audio = torchaudio.functional.resample(wav, orig_freq= sr, new_freq = sample_rate)

    if is_inference:
        p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1] # p: 637
        audio = F.pad(audio, (0, p), mode='constant').data # padding
    return audio.squeeze(), sample_rate


# ============================================================ #
# import amfm_decompy.pYAAPT as pYAAPT
# import amfm_decompy.basic_tools as basic 

# this is for inference ...
def get_yaapt_f0(audio, # wav_re.numpy().astype(np.float64)
                 sr=16000, 
                 interp=False
                 ):
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0) 
        pitch = pYAAPT.yaapt(basic.SignalObj(y_pad, sr), 
                             **{'frame_length': 20.0, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0}
                             )
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)  

# f0_yaapt = get_yaapt_f0(wav_re.numpy().astype(np.float64), sr=16000, interp=False)
# len(f0_yaapt) # 2313



# ============================================================ #
# Normalize F0
def kimino_namaewa_f0_normalize(f0):
    
    f0_x = f0.copy()
    # f0_x= f0_yaapt[0, 0].copy()
    f0_x = torch.log(torch.FloatTensor(f0_x+1))
    ii = f0 != 0

    # @ heiscold (4080)
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std() 

    # @ 4090
    # epsilon = 1e-8
    # if f0[ii].std() == 0:
    #     f0[ii] = (f0[ii] - f0[ii].mean()) / (f0[ii].std() + epsilon)
    # else:
    #     f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std() 

    f0_norm_x = torch.FloatTensor(f0)
    return f0_norm_x

# ============================================================ #

def main(args):

    config_path = args.config

    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    # train - hparams['data']['train_filelist_path']: '/home/heiscold/data/Libritts_r/train_clean_100_360_valid_concat.csv'
    # test - hparams['data']['test_filelist_path']: '/home/heiscold/data/Libritts_r/test_clean_path_text.csv'

    if args.train_or_test == "train":
        df = pd.read_csv(hparams['data']['train_filelist_path'])
        # filelist_path == csv_path: 
        print(df.shape)
    else:
        df = pd.read_csv(hparams['data']['test_filelist_path'])
        # filelist_path == csv_path: 
        print(df.shape)

    audio_paths = df['audio_path'].to_list()
    
    assert df.shape[0] == len(audio_paths)

    bar = tqdm(audio_paths)
    for audio_path in bar:

        # Load & Resample
        audio, sr = load_audio_to_torch(audio_path, is_inference=False)
        # audio.squeeze(), sample_rate

        # Get f0 (1): YAAPT
        f0_yaapt = get_yaapt_f0(audio.unsqueeze(0).numpy().astype(np.float64), sr=16000, interp=False) # f0_yaapt.shape: [1, 1, 2313] # numpy array
        f0 = f0_yaapt[0, 0] # numpy array
        f0_tensor = torch.from_numpy(f0)
        
        # Get f0 (2): pyworld
        # pw_sample_rate = 16000
        # f0, _, _ = pw.wav2world(audio.numpy().astype(np.float64), pw_sample_rate)
        # f0_tensor = torch.from_numpy(f0)

        # Normalize f0
        f0_norm = kimino_namaewa_f0_normalize(f0) # torch.tensor

        if 'train-clean-100' in audio_path:

            # save_file_name
            # '103_1241_000000_000001'
            file_name_unit = audio_path.split('train-clean-100')[1].split('/')[-1].split('.')[0]

            # f0 save:
            save_file_name = file_name_unit + '_f0.pt'
            # '103_1241_000000_000001_f0.pt'
            
            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-100/f0/"
            torch.save(f0_tensor, base_f0_save_path + save_file_name)

            # f0_norm save:
            save_file_name = file_name_unit + '_f0_norm.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_norm_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-100/f0_norm/"
            torch.save(f0_norm, base_f0_norm_save_path + save_file_name)

        elif 'train-clean-360' in audio_path:
            
            # save_file_name
            # '103_1241_000000_000001'
            file_name_unit = audio_path.split('train-clean-360')[1].split('/')[-1].split('.')[0]

            # f0 save:
            save_file_name = file_name_unit + '_f0.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-360/f0/"
            torch.save(f0_tensor, base_f0_save_path + save_file_name)

            # f0_norm save:
            save_file_name = file_name_unit + '_f0_norm.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_norm_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-360/f0_norm/"
            torch.save(f0_norm, base_f0_norm_save_path + save_file_name)

        else:
            # test-clean
            # save_file_name
            # '103_1241_000000_000001'
            file_name_unit = audio_path.split('test-clean')[1].split('/')[-1].split('.')[0]

            # f0 save:
            save_file_name = file_name_unit + '_f0.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/test-clean/f0/"
            torch.save(f0_tensor, base_f0_save_path + save_file_name)

            # f0_norm save:
            save_file_name = file_name_unit + '_f0_norm.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/test-clean/f0_norm/"
            torch.save(f0_norm, base_f0_save_path + save_file_name)

# ============================================================ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config', 
                        type=str, 
                        # required=True,
                        help='JSON file for configuration',
                        default = "/home/heiscold/diff_hier_vc/configs/config_16k.json"
                        )
    
    parser.add_argument('-t',
                        '--train_or_test', 
                        type=str, 
                        # required=True,
                        help='train or test',
                        default = "train",
                        )
    args = parser.parse_args()
    main(args)