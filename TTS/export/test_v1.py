from  kkcode.kkTools import calcTools
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description="singlecodec2_test")
parser.add_argument('--FAKE_DIR')
parser.add_argument('--REAL_DIR')
parser.add_argument('--OUT_DIR')
parser.add_argument('--SAMPLE_RATE')
parser.add_argument('--DEVICE')
parser.add_argument('--NUMTHREAD')

args = parser.parse_args()

NUMTHREAD = int(args.NUMTHREAD)
SAMPLE_RATE = int(args.SAMPLE_RATE)
DEVICE = args.DEVICE
FAKE_DIR = args.FAKE_DIR
REAL_DIR = args.REAL_DIR
OUT_DIR = args.OUT_DIR

os.makedirs(OUT_DIR, exist_ok=True)
sub_dir = '/'.join(FAKE_DIR.split('/')[-4:])
out_item_dir = os.path.join(OUT_DIR, sub_dir)
os.makedirs(out_item_dir, exist_ok=True)

out_text_path = out_item_dir+'/'+"res.txt"

# 定义常量
# NUMTHREAD = 1  # 替换为实际的线程数
# SAMPLE_RATE = 16000
# DEVICE = "cuda:0"
# FAKE_DIR = '/10.170.28.44/nwpu_tts/jbhu/codec/kkrun_zip/output/wav/codec/SingleCodec2/v1_emilia_10wh/syn_he30w'
# REAL_DIR = "/10.170.28.44/nwpu_tts/jbhu/codec/kkrun_zip/testset/test/epressive"
# OUT_DIR = ''


UTTS = []  # 替换为实际的音频文件名列表
LANGUAGE_JUDGE = "chinese"  # 替换为实际的语言判断
# LANGUAGE_JUDGE = "english"  # 替换为实际的语言判断


STEPS = [6]
# STEPS = [5,6,8]
# STEPS = [8]




for utt in os.listdir(FAKE_DIR):
    UTTS.append(utt.split(".")[0])

with open(out_text_path,'w') as f:
    if 2 in STEPS:
        # stoi
        pesq = calcTools.STOI(sample_rate=SAMPLE_RATE, device=DEVICE)
        result = pesq.run(FAKE_DIR, REAL_DIR, UTTS, use_tqdm=False, numthread=NUMTHREAD)        
        print('Stoi ', round(np.mean(np.array(result)), 3), np.mean(np.array(result)))
        f.write(f'Stoi:{round(np.mean(np.array(result)), 3), np.mean(np.array(result))}\n')

    if 3 in STEPS:
        # pesq
        pesq = calcTools.PESQ(sample_rate=SAMPLE_RATE, device=DEVICE)
        result = pesq.run(FAKE_DIR, REAL_DIR, UTTS, use_tqdm=False, numthread=NUMTHREAD)        
        print('Pesq ', round(np.mean(np.array(result)), 3))
        f.write(f'Pesq:{round(np.mean(np.array(result)), 3)}\n')

    if 4 in STEPS:
        # mcd
        mcd = calcTools.MCD(sample_rate=SAMPLE_RATE)
        result = mcd.run(FAKE_DIR, REAL_DIR, UTTS, use_tqdm=False, numthread=NUMTHREAD)
        print('MCD ', round(np.mean(np.array(result)), 3))
        f.write(f'MCD:{round(np.mean(np.array(result)), 3)}\n')

    if 5 in STEPS:
        # speech mos
        mos = calcTools.SpeechMOS(sample_rate=SAMPLE_RATE, device=DEVICE)
        result = mos.run(FAKE_DIR, UTTS, use_tqdm=False, numthread=NUMTHREAD)
        print('SpeechMOS ', round(np.mean(np.array(result)), 3))
        f.write(f'SpeechMOS:{round(np.mean(np.array(result)), 3)}\n')

    if 6 in STEPS:
        # wespeaker
        wespeaker = calcTools.WespeakerCalc(DEVICE,LANGUAGE_JUDGE)
        result = wespeaker.run(FAKE_DIR, REAL_DIR, UTTS, use_tqdm=False, numthread=NUMTHREAD)        
        print('wespeaker Simi', round(np.mean(np.array(result)), 3))
        f.write(f'wespeaker Simi:{round(np.mean(np.array(result)), 3)}\n')

    if 7 in STEPS:
        # copysyn 说话人和原始说话人的相似度
        sim = calcTools.WavLmSvCalc(DEVICE,LANGUAGE_JUDGE)
        result = sim.run(FAKE_DIR, REAL_DIR, UTTS)        
        print('wavlm Simi', round(np.mean(np.array(result)), 3))
        f.write(f'wavlm Simi:{round(np.mean(np.array(result)), 3)}\n')

    if 8 in STEPS:
        # copysyn 相对wer值
        wer = calcTools.WER(DEVICE,LANGUAGE_JUDGE)
        result, utt2content = wer.run(FAKE_DIR, REAL_DIR, UTTS)        
        # result = wespeaker.run(FAKE_DIR, REAL_DIR, UTTS, use_tqdm=False, numthread=NUMTHREAD)        
        print('copysyn wer', round(np.mean(result), 3))
        f.write(f'WER:{round(np.mean(result), 3)}\n')

        for utt, item in sorted(utt2content.items(), key=lambda x: x[1]['score'], reverse=True):
            f.write(f'\tUTT:{utt} WER:{item["score"]}\n\tcontent_gt:{item["content_gt"]}\n\tcontent_pred:{item["content_pred"]}\n\tS:{item["S"]}   D:{item["D"]}   I:{item["I"]}\n\n\n')
