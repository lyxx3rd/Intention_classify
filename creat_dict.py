import json

path = "./data/IMCS-DAC/IMCS-DAC_train.json"
data = json.load(open(path, encoding='utf8'))
keys_list = list(data.keys())
dialogue_act_list = []
n = 0
for key in keys_list:
    for i in range(len(data[key])):
        dialogue_act_list.append(data[key][i]['dialogue_act'])
        n += 1
unique_list = list(set(dialogue_act_list))
label_dict = {str(i): value for i, value in enumerate(unique_list)}
reversed_label_dict = {value: key for key, value in label_dict.items()}

with open('./data/id2word.json', 'w', encoding='utf8') as file:
    json.dump(label_dict, file)
with open('./data/word2id.json', 'w', encoding='utf8') as file:
    json.dump(reversed_label_dict, file)