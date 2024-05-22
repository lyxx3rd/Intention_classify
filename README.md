# Intention_classify
# 意图识别算法

## 流程：
创建字典-处理数据-训练模型-推理预测结果

## 安装运行环境
运行
```{bash}
pip install -r requirements.txt
```

## 创建字典
下载数据，将数据放入"./data/"目录中。

运行
```{bash}
python creat_dict.py
```

## 处理数据

运行
```{bash}
python data_precoee.py
```

## 训练模型
下载预训练分类模型，放入"./Model/"中。
创建文件夹"./Model_save/"
例如：git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext

运行
```{bash}
python train.py
```
此时训练好会保存在"./Model_save/"中。

## 预测结果
修改"dev.py"中“model = torch.load("./Model_save/classify_0.85400.pt")”部分的模型名称，以上述步骤实际生成模型名为准。
创建文件夹“./outputs/”

运行
```{bash}
python dev.py
```
生成结果会保存在“./outputs/”文件夹中。

## 备注
### 注1.
如数据结构与本实例结构不同，则在创建字典、处理数据和预测结果等环节需重新处理代码
### 注2.
如创建文件夹名称与上述不一致，请在原码中修改文件路径
