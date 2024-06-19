# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

import julius
import torch
from torch import nn
from torch.nn import functional as F

from .states import capture_init
from .utils import center_trim, unfold
from .transformer import LayerScale

from mamba_ssm import Mamba
from mamba_ssm import Mamba2

import logging

logger = logging.getLogger(__name__)

def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)

class MambaWrapper(nn.Module):
    """
    Wrapper for Mamba.
    """
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        #self.linear = nn.Linear(self.num_directions * dim, dim)
        self.conv = nn.Conv1d(dim, dim, 1)
        self.norm_fn = nn.GroupNorm(1, dim)
        self.act = nn.GELU()

    def forward(self, x):
        #logger.info(x.shape)
        x = x.permute(0, 2, 1)
        #logger.info('After permute')
        #logger.info(x.shape)

        x = self.mamba(x)
        #logger.info('After mamba')
        #logger.info(x.shape)
        #x = x.permute(1, 0, 2)
        #x = self.linear(x)
        #logger.info('After linear')
        #logger.info(x.shape)
        #x = x.permute(1, 2, 0)
        x = x.permute(0, 2, 1)
        #logger.info('After permute 2')
        #logger.info(x.shape)
        #assert False
        x = self.conv(x)
        x = self.act(x)
        x = self.norm_fn(x)
        return x
    
class BiMambaWrapper(nn.Module):
    """
    Bidirectional Mamba.
    """
    def __init__(self, dim):
        super().__init__()
        self.mamba_forw = Mamba(
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.mamba_back = Mamba(
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.conv = nn.Conv1d(2 * dim, 2 * dim, 1)
        self.act = nn.GLU(1)
        self.norm_fn = nn.GroupNorm(1, dim)


    def forward(self, x):
        x = x.permute(0, 2, 1)

        x_back = x.flip(dims=(1,))

        x = self.mamba_forw(x)

        x_back = self.mamba_back(x_back)

        x = torch.cat((x, x_back), dim=2)
        x = x.permute(0, 2, 1)
        
        x = self.conv(x)
        x = self.act(x)
        x = self.norm_fn(x)
        
        return x


class Drumamba(nn.Module):
    @capture_init
    def __init__(self,
                 sources,
                 # Channels
                 audio_channels=2,
                 channels=64,
                 growth=2.,
                 # Main structure
                 depth=6,
                 rewrite=True,
                 mamba_layers=0,
                 bi_mamba=False,
                 # Convolutions
                 kernel_size=8,
                 stride=4,
                 context=1,
                 # Activations
                 gelu=True,
                 glu=True,
                 # Normalization
                 norm_starts=4,
                 norm_groups=4,
                 # Pre/post processing
                 normalize=True,
                 resample=True,
                 # Weight init
                 rescale=0.1,
                 # Metadata
                 samplerate=44100,
                 segment=4 * 10):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            mamba_layers (int): number of mamba layers, 0 = no lstm. Deactivated
                by default, as this is now replaced by the smaller and faster small LSTMs
                in the DConv branches.
            bi_mamba (bool): use bidirectional mamba instead of simple mamba.
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time steps.
            gelu: use GELU activation function.
            glu (bool): use glu instead of ReLU for the 1x1 rewrite conv.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            normalize (bool): normalizes the input audio on the fly, and scales back
                the output by the same amount.
            resample (bool): upsample x2 the input and downsample /2 the output.
            rescale (float): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment (float): duration of the chunks of audio to ideally evaluate the model on.
                This is used by `demucs.apply.apply_model`.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment = segment
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_scales = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        if gelu:
            act2 = nn.GELU
        else:
            act2 = nn.ReLU

        in_channels = audio_channels
        padding = 0
        for index in range(depth):
            norm_fn = lambda d: nn.Identity()  # noqa
            if index >= norm_starts:
                norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa

            encode = []
            encode += [
                nn.Conv1d(in_channels, channels, kernel_size, stride),
                act2(),
                norm_fn(channels),
            ]
            if rewrite:
                encode += [
                    nn.Conv1d(channels, ch_scale * channels, 1),
                    activation, norm_fn(channels)]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
            if rewrite:
                decode += [
                    nn.Conv1d(channels, ch_scale * channels, 2 * context + 1, padding=context),
                    activation, norm_fn(channels)]
            decode += [nn.ConvTranspose1d(channels, out_channels,
                       kernel_size, stride, padding=padding)]
            if index > 0:
                decode += [act2(), norm_fn(out_channels)]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        if mamba_layers:
            self.mamba = BiMambaWrapper(channels) if bi_mamba else MambaWrapper(channels)
        else:
            self.mamba = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolution, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        Note that input are automatically padded if necessary to ensure that the output
        has the same length as the input.
        """
        if self.resample:
            length *= 2

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        if self.mamba:
            x = self.mamba(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            skip = center_trim(skip, x)
            x = decode(x + skip)

        if self.resample:
            x = julius.resample_frac(x, 2, 1)
        x = x * std + mean
        x = center_trim(x, length)
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x

    def load_state_dict(self, state, strict=True):
        # fix a mismatch with previous generation Demucs models.
        for idx in range(self.depth):
            for a in ['encoder', 'decoder']:
                for b in ['bias', 'weight']:
                    new = f'{a}.{idx}.3.{b}'
                    old = f'{a}.{idx}.2.{b}'
                    if old in state and new not in state:
                        state[new] = state.pop(old)
        super().load_state_dict(state, strict=strict)
