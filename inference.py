import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import soundfile as sf



def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

config_path = "configs/config.json" #@param {type:"string"}
model_path = "G_2000.pth" #@param {type:"string"}
hps = utils.get_hparams_from_file(config_path)
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
model = net_g.eval()
model = utils.load_checkpoint(model_path, net_g, None)

speaker_id = 90 #@param {type:"number"}
text = "あなたに会えて嬉しいですわ!にゃーん。" #@param {type:"string"}
noise_scale=0.6 #@param {type:"number"}
noise_scale_w=0.668 #@param {type:"number"}
length_scale=1 #@param {type:"number"}
stn_tst = get_text(text, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()


sf.write('audio.wav', audio, hps.data.sampling_rate)
