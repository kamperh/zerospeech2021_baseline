#!/usr/bin/env python

"""
Get individual wav files for an indicated dataset and split.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import librosa
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("input_dir", type=str, help="local copy of data")
    parser.add_argument("dataset", type=str, help="e.g. 'zs2017_fr'")
    parser.add_argument("split", type=str, help="e.g. 'train'")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    input_dur = Path(args.input_dir)

    json_fn = (Path("datasets")/args.dataset/args.split).with_suffix(".json")
    print(f"Reading: {json_fn}")
    with open(json_fn) as f:
        metadata = json.load(f)
        for in_path, start, duration, out_path in tqdm(metadata):
            wav_path = input_dur/in_path
            assert wav_path.with_suffix(".wav").exists(), (
                "'{}' does not exist".format(
                wav_path.with_suffix(".wav"))
                )
            out_path = Path("wav/")/args.dataset/args.split/Path(out_path).stem
            out_path.parent.mkdir(parents=True, exist_ok=True)
            wav, _ = librosa.load(
                wav_path.with_suffix(".wav"), sr=16000,
                offset=start, duration=duration
                )
            librosa.output.write_wav(
                out_path.with_suffix(".wav"), wav, sr=16000
                )


if __name__ == "__main__":
    main()
