#!/usr/bin/env python

"""
Encodes a given dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import numpy as np
import os
import sys
import torch

from cpc.dataset import findAllSeqs
from cpc.feature_loader import buildFeature, buildFeature_batch
from scripts.utils.utils_functions import (
    readArgs, writeArgs, loadCPCFeatureMaker, loadClusterModule
    )


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "dataset_dir", type=str, help="Path to dataset directory"
        )
    parser.add_argument(
        "out_dir", type=str, help="Path to dataset directory"
        )
    parser.add_argument(
        "--output_format",
        choices=["npy", "txt"], default="txt"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def encode_dataset(args):
    """Written by Benji."""
    dataset_dir, out_dir = Path(args.dataset_dir), Path(args.out_dir)
    output_format = args.output_format
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"indices").mkdir(parents=True, exist_ok=True)
    (out_dir/"auxiliary_embedding2").mkdir(parents=True, exist_ok=True)

    clustering_args = readArgs(
        "checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50_args.json"
        )
    clusterModule = loadClusterModule(
        "checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt"
        )
    clusterModule.cuda()

    # Maybe it's relative path
    if not os.path.isabs(clustering_args.pathCheckpoint):
        clustering_args.pathCheckpoint = os.path.join(
            os.path.dirname(os.path.abspath(
            "checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt")),
            clustering_args.pathCheckpoint
            )
    assert os.path.exists(clustering_args.pathCheckpoint), (
        f"CPC path at {clustering_args.pathCheckpoint} does not exist!!"
        )

    featureMaker = loadCPCFeatureMaker(
        clustering_args.pathCheckpoint, 
        gru_level=vars(clustering_args).get('level_gru', None), 
        get_encoded=clustering_args.encoder_layer, 
        keep_hidden=True)
    featureMaker.eval()
    featureMaker.cuda()

    codebook = clusterModule.Ck.squeeze().cpu().numpy()
    fn = out_dir/"embedding.npy"
    print(codebook.shape)
    print("Writing: {}".format(fn))
    np.save(fn, codebook)

    def cpc_feature_function(x): 
        return buildFeature(
            featureMaker, x, seqNorm=False, strict=True, maxSizeSeq=10240
            )


    # for path in tqdm(sorted(list(dataset_dir.rglob("*.flac")))):
    for path in tqdm(sorted(list(dataset_dir.rglob("*.wav")))):

        # # Temp
        # if (out_dir/"indices"/path.stem).with_suffix(".txt").exists():
        #     continue
        # print(path, output_format)

        out_path = (out_dir / path.stem).with_suffix("")
        out_path.parent.mkdir(exist_ok=True, parents=True)

        features = cpc_feature_function(path).cuda()
        codes = torch.argmin(clusterModule(features), dim=-1)
        codes = codes[0].detach().cpu().numpy()

        features = features[0].detach().cpu().numpy()

        one_hot = np.zeros((len(codes), 50), dtype=int)
        one_hot[np.arange(len(codes)), codes] = 1

        if output_format == "txt":
            with open(out_path.with_suffix(".txt"), "w") as file:
                np.savetxt(file, one_hot, fmt="%d")

        out_path = out_dir/"indices"/path.stem
        if output_format == "txt":
            with open(out_path.with_suffix(".txt"), "w") as file:
                np.savetxt(file, codes, fmt="%d")
        elif output_format == "npy":
            np.save(out_path.with_suffix(".npy"), codes)

        out_path = out_dir/"auxiliary_embedding2"/path.stem
        if output_format == "txt":
            with open(out_path.with_suffix(".txt"), "w") as file:
                np.savetxt(file, features, fmt="%.16f")
        elif output_format == "npy":
            np.save(out_path.with_suffix(".npy"), features)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    encode_dataset(args)
    


if __name__ == "__main__":
    main()
