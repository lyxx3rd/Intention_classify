"""
本文件是用来生成分类数据的。
每一个query，生成两个例子，一个是正例，另一个是负例。负例从接口中调取。
"""
import numpy as np
import json
import os.path
import random
from dataclasses import dataclass, asdict
from typing import Optional

from loguru import logger
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput

logger.add("out.log")
import numpy
import requests
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./Model/chinese-roberta-wwm-ext"

class DatasetClassify(Dataset):
    def __init__(self, path):
        self.data_list = json.load(open(path, encoding='utf8'))

    def __getitem__(self, index):
        item = self.data_list[index]
        item = DataItem(**item)
        content = item.history + '[SEP]' + item.sentence
        return content, item.label

    def __len__(self):
        return len(self.data_list)


def collator_fn(batch):
    batch = numpy.array(batch)

    data_batch = batch[:, 0]
    label_batch = numpy.array(batch[:, 1], dtype=int)
    data_batch = tokenizer(data_batch.tolist(), max_length=256, padding=True, truncation=True,
                           return_tensors="pt").to(DEVICE)
    return data_batch, torch.tensor(label_batch, device=DEVICE, dtype=torch.long)


@dataclass
class DataItem:
    history: str
    sentence: str
    label: int


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = torch.load("./Model_save/classify_model.pt")
model.to(DEVICE)
test_data_loader = DataLoader(DatasetClassify("./data/IMCS-DAC/test_list_data.json"), batch_size=32, shuffle=False,
                             collate_fn=collator_fn)
model.eval()
pred_label = []
for item, label in tqdm(test_data_loader, position=0, leave=True):
    model.eval()
    output = model(**item)
    pre_label = output.logits.detach().cpu().numpy()
    pre_label = np.argmax(pre_label, axis=1)
    pred_label.append(pre_label)

flattened_list = [element for sublist in pred_label for element in sublist]
data = json.load(open("./data/IMCS-DAC/IMCS-DAC_test.json", encoding='utf8'))
label_dict = json.load(open("./id2word.json", encoding='utf8'))
keys_list = list(data.keys())
n = 0
for key in keys_list:
    for i in range(len(data[key])):
        data[key][i]['dialogue_act'] = label_dict[str(flattened_list[n])]
        n += 1
data[keys_list[0]]

filename = f"./outputs/IMCS-DAC_test.json"

# 使用json.dump()函数将列表写入文件
with open(filename, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False)
