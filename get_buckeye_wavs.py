#!/usr/bin/env python

"""
Extract individual wav files for Buckeye.

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
        "buckeye_dir", type=str, help="local copy of the official Buckeye data"
        )
    parser.add_argument(
        "--segments", action="store_true",
        help="read Buckeye segments instead of the full utterances"
        )
    parser.add_argument(
        "--felix", action="store_true",
        help="use the Felix Kreuk Buckeye splits"
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

    print(args.segments)

    in_dir = Path(args.buckeye_dir)
    if args.segments:
        out_dir = Path("datasets/buckeye_segments/")
    elif args.felix:
        out_dir = Path("datasets/buckeye_felix/")
    else:
        out_dir = Path("datasets/buckeye/")

    # for split in ["train", "test", "val"]:
    # for split in ["val", "test"]:
    for split in ["test"]:
        print("Extracting utterances for {} set".format(split))

        split_path = out_dir / split
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
                if args.segments:
                    out_path = Path("wav/buckeye_segments/")/split/Path(
                        out_path
                        ).stem
                elif args.felix:
                    out_path = Path("wav/buckeye_felix/")/split/Path(
                        out_path
                        ).stem
                else:
                    out_path = Path("wav/buckeye/")/split/Path(out_path).stem
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
