#!/usr/bin/env python

"""
Encodes a given dataset using HuBERT.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2023
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "wav_dir", type=str, help="Audio directory"
        )
    parser.add_argument(
        "output_dir", type=str, help="Output directory"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Directories
    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"indices").mkdir(parents=True, exist_ok=True)
    (output_dir/"prequant").mkdir(parents=True, exist_ok=True)
    (output_dir/"layer09").mkdir(parents=True, exist_ok=True)
    # (output_dir/"layer10").mkdir(parents=True, exist_ok=True)

    # Model
    hubert = torch.hub.load(
        "bshall/hubert:main", "hubert_discrete", trust_repo=True
    ).cuda()

    # Embedding
    codebook = hubert.kmeans.cluster_centers_
    output_fn = output_dir/"embedding.npy"
    print("Codebook:", codebook.shape)
    print("Writing:", output_fn)
    np.save(output_fn, codebook)

    n_min_samples = 640
    n_max_samples = 2000000
    print("Writing to:", output_dir)
    for i, wav_fn in enumerate(tqdm(sorted(list(wav_dir.rglob("*.wav"))))):
    # for wav_fn in ["wav/buckeye/val/s18_01b_004469-004474.wav",]:

        # Input
        wav, sr = torchaudio.load(wav_fn)
        assert sr == 16000
        l = wav.shape[1]

        if l > n_max_samples:
            wav = wav[:, :n_max_samples]
        elif l < n_min_samples:
            wav = F.pad(wav, (0, n_min_samples - l))
        x = wav.unsqueeze(0).cuda()
        x = F.pad(x, ((400 - 320) // 2, (400 - 320) // 2))

        # Extract features
        prequant, _ = hubert.encode(x, layer=7)
        prequant = prequant.detach().squeeze().cpu().numpy()

        # Higher layers
        layer09, _ = hubert.encode(x, layer=9)
        layer09 = layer09.detach().squeeze().cpu().numpy()
        # layer10, _ = hubert.encode(x, layer=10)
        # layer10 = layer10.detach().squeeze().cpu().numpy()

        # Extract units
        units = hubert.kmeans.predict(prequant)

        # # Write features
        output_fn = (output_dir/"prequant"/wav_fn.stem).with_suffix(".npy")
        np.save(output_fn, prequant)
        output_fn = (output_dir/"layer09"/wav_fn.stem).with_suffix(".npy")
        np.save(output_fn, layer09)
        # output_fn = (output_dir/"layer10"/wav_fn.stem).with_suffix(".npy")
        # np.save(output_fn, layer10)

        # Write units
        output_fn = (output_dir/"indices"/wav_fn.stem).with_suffix(".txt")
        np.savetxt(output_fn, units, fmt="%d")


if __name__ == "__main__":
    main()
