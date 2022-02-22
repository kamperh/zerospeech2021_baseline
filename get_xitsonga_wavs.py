#!/usr/bin/env python

"""
Extract individual wav files for Xitsonga.

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
    parser.add_argument(
        "data_dir", type=str, help="local copy of the official data directory"
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

    in_dir = Path(args.data_dir)
    out_dir = Path("datasets/xitsonga/")

    # for split in ["train", "test", "val"]:
    for split in ["train"]:
        print("Extracting utterances for {} set".format(split))

        split_path = out_dir/split
        if not split_path.with_suffix(".json").exists():
            print("Skipping {} (no json file)".format(split))
            continue
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            for in_path, start, duration, out_path in tqdm(metadata):
                wav_path = in_dir/in_path
                assert wav_path.with_suffix(".wav").exists(), (
                    "'{}' does not exist".format(
                    wav_path.with_suffix(".wav"))
                    )
                out_path = Path("wav/xitsonga/")/split/Path(out_path).stem
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
