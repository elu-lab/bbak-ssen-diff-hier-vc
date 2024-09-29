import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
# from module.utils import parse_filelist
from torch.nn import functional as F
np.random.seed(1234)


# ==================================================== #

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


# ==================================================== #

# Original: [module > utils.py]
# def parse_filelist(filelist_path):
#     with open(filelist_path, 'r') as f:
#         filelist = [line.strip() for line in f.readlines()]
#     return filelist

def parse_filelist(filelist_path):
    # filelist_path == csv_path: 
    # - train: '/home/heiscold/data/Libritts_r/train_clean_100_360_valid_concat.csv'
    # - test: '/home/heiscold/data/Libritts_r/test_clean_path_text.csv'

    temp_df = pd.read_csv(filelist_path)
    filelist = temp_df['audio_path'].to_list()
    return filelist

# ==================================================== #

def convert_filename(audio_path):

        if 'train-clean-100' in audio_path:

            # save_file_name
            # '103_1241_000000_000001'
            file_name_unit = audio_path.split('train-clean-100')[1].split('/')[-1].split('.')[0]

            # f0 save:
            save_file_name = file_name_unit + '_f0.pt'
            # '103_1241_000000_000001_f0.pt'
            
            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-100/f0/"
            f0_path = base_f0_save_path + save_file_name

            # f0_norm save:
            save_file_name = file_name_unit + '_f0_norm.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_norm_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-100/f0_norm/"
            f0_norm_path = base_f0_norm_save_path + save_file_name

        elif 'train-clean-360' in audio_path:
            
            # save_file_name
            # '103_1241_000000_000001'
            file_name_unit = audio_path.split('train-clean-360')[1].split('/')[-1].split('.')[0]

            # f0 save:
            save_file_name = file_name_unit + '_f0.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-360/f0/"
            f0_path = base_f0_save_path + save_file_name

            # f0_norm save:
            save_file_name = file_name_unit + '_f0_norm.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_norm_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/train-clean-360/f0_norm/"
            f0_norm_path = base_f0_norm_save_path + save_file_name

        else:
            # test-clean
            # save_file_name
            # '103_1241_000000_000001'
            file_name_unit = audio_path.split('test-clean')[1].split('/')[-1].split('.')[0]

            # f0 save:
            save_file_name = file_name_unit + '_f0.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/test-clean/f0/"
            f0_path = base_f0_save_path + save_file_name

            # f0_norm save:
            save_file_name = file_name_unit + '_f0_norm.pt'
            # '103_1241_000000_000001_f0.pt'

            base_f0_save_path = f"/home/heiscold/diff_hier_vc/preprocessed/test-clean/f0_norm/"
            f0_norm_path = base_f0_norm_save_path + save_file_name

        return f0_path, f0_norm_path

# ==================================================== #

def parse_f0_lists(filelist_path):
    # filelist_path == csv_path: 
    # - train: '/home/heiscold/data/Libritts_r/train_clean_100_360_valid_concat.csv'
    # - test: '/home/heiscold/data/Libritts_r/test_clean_path_text.csv'

    temp_df = pd.read_csv(filelist_path)
    filelist = temp_df['audio_path'].to_list()

    filelist_f0, filelist_f0_norm = [], []
    for audio_path in filelist:

        # Convert F0 Path
        f0_path, f0_norm_path = convert_filename(audio_path)

        # f0_path, f0_norm_path lists
        filelist_f0.append(f0_path)
        filelist_f0_norm.append(f0_norm_path)

    return filelist_f0, filelist_f0_norm 

# ==================================================== #

class AudioDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """
    def __init__(self, config, training=True):
        super(AudioDataset, self).__init__()
        self.config = config
        self.hop_length = config.data.hop_length
        self.training = training
        self.mel_length = config.train.segment_size // config.data.hop_length
        self.segment_length = config.train.segment_size
        self.sample_rate = config.data.sampling_rate

        # train_filelist_path: "fp_16k/train_wav.txt" -> '/home/heiscold/data/Libritts_r/train_clean_100_360_valid_concat.csv'
        # test_filelist_path: "fp_16k/test_wav.txt" -> '/home/heiscold/data/Libritts_r/test_clean_path_text.csv'
        self.filelist_path = config.data.train_filelist_path if self.training else config.data.test_filelist_path
        self.audio_paths = parse_filelist(self.filelist_path) if self.training else parse_filelist(self.filelist_path)[:101] 

        self.f0_norm_paths, self.f0_paths  = parse_f0_lists(self.filelist_path)

    def load_audio_to_torch(self, audio_path):
        # torchaudio load
        wav, sr = torchaudio.load(audio_path, 
                                #   normalize=True (Default)
                                )

        # resample
        sample_rate = 16000
        audio = torchaudio.functional.resample(wav, orig_freq= sr, new_freq = sample_rate)

        if not self.training:
            p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1] # p: 637
            audio = F.pad(audio, (0, p), mode='constant').data # padding
        return audio.squeeze(), sample_rate

        # if self.training:
        # - audio.shape: (torch.Size([1, 184963])
        # - audio.squeeze().shape: torch.Size([184963]))

        # if not self.training:
        # - p: 637
        # - audio: [1, 185600]

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        f0_norm_path = self.f0_norm_paths[index]
        f0_path = self.f0_paths[index]

        audio, sample_rate = self.load_audio_to_torch(audio_path)
        # torch.Size([184963]), 16000
        
        # 환장하겠네... f0, f0_norm 공개를 안 해놨네...?
        # 함수 하나하나 짜야하는 건가...?
        # 아오....
        # 누가 이기나 보자. 
        f0_norm = torch.load(f0_norm_path)
        f0 = torch.load(f0_path)

        # Convert to Pytorch Tensor + Matching Shapes
        f0_norm = f0_norm.unsqueeze(0)
        f0 = f0.unsqueeze(0)
        # f0 = torch.from_numpy(f0).unsqueeze(0)
        # torch.Size([1, 2313]), torch.Size([1, 2313])

        assert sample_rate == self.sample_rate, \
            f"""Got path to audio of sampling rate {sample_rate}, \
                but required {self.sample_rate} according config."""
        
        if not self.training:  

            # return audio, f0_norm, f0
            # wav_re.shape, f0_norm_x.shape, f0(=f0_new).shape[-1]
            # torch.Size([184963]), torch.Size([2313]), 2313 # f0: numpy array
            
            f0_norm = f0_norm.unsqueeze(0)
            f0 = torch.from_numpy(f0).unsqueeze(0)
            return audio, f0_norm, f0
            # torch.Size([184963]), torch.Size([1, 2313]), torch.Size([1, 2313])

        # segment_length = 35840
        if audio.shape[-1] > self.segment_length:
            # True
            max_f0_start = f0.shape[-1] - self.segment_length//80 
            # max_f0_start : 1865
            f0_start = np.random.randint(0, max_f0_start)
            # f0_start: 762

            f0_norm_seg = f0_norm[:, f0_start:f0_start + self.segment_length // 80]  
            # f0_norm_seg: torch.Size([1, 448])

            f0_seg = f0[:, f0_start:f0_start + self.segment_length // 80]  
            # f0_seg: torch.Size([1, 448])

            audio_start = f0_start*80
            # audio_start: 60960

            segment = audio[audio_start:audio_start + self.segment_length]
            # segment.shape: torch.Size([35840])

            # hop_length = 320
            # segment_length = 35840
            # mel_length= segment_length // hop_length
            # mel_length: 112
            if segment.shape[-1] < self.segment_length:
                segment = F.pad(segment, (0, self.segment_length - segment.shape[-1]), 'constant') 

            length = torch.LongTensor([self.mel_length])
            # length: tensor([112])
        
        else:
            segment = F.pad(audio, (0, self.segment_length - audio.shape[-1]), 'constant') 
            length = torch.LongTensor([audio.shape[-1] // self.hop_length])

            f0_norm_seg = F.pad(f0_norm, (0, self.segment_length // 80 - f0_norm.shape[-1]), 'constant') 
            
            f0_seg = F.pad(f0, (0, self.segment_length // 80 - f0.shape[-1]), 'constant') 

        return segment, f0_norm_seg, f0_seg, length
        # segment: torch.Size([35840])
        # f0_norm_seg: torch.Size([1, 448])
        # f0_seg: torch.Size([1, 448])
        # length: tensor([112])

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch
