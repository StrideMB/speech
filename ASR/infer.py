import os
import pandas as pd
from funasr import AutoModel
from jiwer import cer  # pip install jiwer

# 加载模型
model = AutoModel(model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")#引号中填下载模型的文件夹

# 读取 CSV 文件

csv_file = "speech_asr_aishell_testsets_part.csv" # S0764,S0765,S0766,S0767
df = pd.read_csv(csv_file)

preds = []
cers = []

for idx, row in df.iterrows():
    audio_path = row['Audio:FILE']
    ref_text = row['Text:LABEL'].replace(" ", "")  
    try:
        res = model.generate(input=audio_path)
        if isinstance(res, list):
            hyp_text = res[0]['text'].replace(" ", "")
        else:
            hyp_text = res['text'].replace(" ", "")

        preds.append(hyp_text)
        cer_val = cer(ref_text, hyp_text)
        cers.append(cer_val)

        print(f"{audio_path}\n参考: {ref_text}\n识别: {hyp_text}\nCER: {cer_val:.4f}\n")

    except Exception as e:
        print(f"识别失败: {audio_path}, 错误: {e}")
        preds.append("")
        cers.append(None)  # 出错则 CER 填 None

# 保存结果到 CSV

df['Predicted'] = preds
df['CER_per_file'] = cers
output_csv = "test_list_with_pred_cer.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"识别结果已保存到: {output_csv}")

#  计算整体 CER

valid_indices = [i for i, c in enumerate(cers) if c is not None]
overall_refs = [df['Text:LABEL'][i].replace(" ", "") for i in valid_indices]
overall_hyps = [preds[i] for i in valid_indices]

overall_cer = cer(overall_refs, overall_hyps)
print(f"\n整体 Character Error Rate (CER): {overall_cer:.4f}")
refs = df['Text:LABEL'].str.replace(" ", "")
hypotheses = df['Predicted'].str.replace(" ", "")
cer_score = cer(refs.tolist(), hypotheses.tolist())

print(f"\n整体 Character Error Rate (CER): {cer_score:.4f}")
