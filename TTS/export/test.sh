#!/bin/bash

python test_v1.py \
    --FAKE_DIR '/home/zxh/speech/TTS/results' \
    --REAL_DIR '/home/zxh/speech/TTS/seedtts_testset/zh/wavs' \
    --OUT_DIR '/home/zxh/speech/TTS/' \
    --DEVICE 'cuda:0' \
    --SAMPLE_RATE 16000 \
    --NUMTHREAD 1 \
    # &
