# Quantized CPC Features for Buckeye


## VQWordSeg-formatted output

Extract individual wav files:

    ./get_buckeye_wavs.py ~/endgame/datasets/buckeye/
    ./get_xitsonga_wavs.py ~/endgame/datasets/zerospeech2015/xitsonga_wavs/

Extract individual segment WAV files for Buckeye:

    ./get_buckeye_wavs.py --segments ~/endgame/datasets/buckeye/

Encode the Buckeye dataset:

    conda activate zerospeech2021_baseline
    ./encode.py wav/buckeye/val/ exp/buckeye/val/
    ./encode.py wav/buckeye/test/ exp/buckeye/test/

Encode with normalization:

    ./encode_normalized.py wav/buckeye/val/ exp/buckeye_normalized/val/


## ZeroSpeech2021-formatted output (deprecated)

    python ./scripts/quantize_audio.py \
        checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
        wav/buckeye/val/ \
        exp/cpc_big/buckeye/val/ \
        --file_extension wav



# Quantized CPC features for ZeroSpeech'17

Extract individual wav files for French ZeroSpeech'17 training data:

    ./get_wavs.py /home/kamperh/endgame/datasets/zerospeech2020/2020/2017/ zs2017_fr train

Encode the dataset:

    conda activate zerospeech2021_baseline
    ./encode.py wav/zs2017_fr/train/ exp/zs2017_fr/train/
