import json
## train/dev/test
type_list = ["train","dev","test"]
for data_type in type_list:
    path = f"./data/IMCS-DAC/IMCS-DAC_{data_type}.json"
    data = json.load(open(path, encoding='utf8'))
    keys_list = list(data.keys())

    label_dict = json.load(open("./data/id2word.json", encoding='utf8'))
    reversed_label_dict = json.load(open("./data/word2id.json", encoding='utf8'))

    data_dumm = data
    for key in keys_list:
        for i in range(len(data_dumm[key])):
            if data[key][i]['dialogue_act'] == '':
                data_dumm[key][i]['dialogue_act'] = "0"
            else :
                data_dumm[key][i]['dialogue_act'] = reversed_label_dict[data[key][i]['dialogue_act']]
    data_dumm_history = {}
    data_list = []
    for key in keys_list:
        data_dumm_history[key]={}
        for i in range(len(data_dumm[key])):
            ## 空字典
            data_dumm_history[key][i]={}
            ## 历史对话
            if i == 0:
                data_dumm_history[key][i]['history'] = "历史记录: \n "
            elif i == 1:
                data_dumm_history[key][i]['history'] = "历史记录: \n " + data_dumm[key][i-1]["sentence_id"] + ". " + str(data[key][i-1]["speaker"]) +" : " + str(data[key][i-1]["sentence"]) + " \n "
            elif i == 2:
                data_dumm_history[key][i]['history'] = "历史记录: \n " + data_dumm[key][i-2]["sentence_id"] + ". " + str(data[key][i-2]["speaker"]) +" : " + str(data[key][i-2]["sentence"]) + " \n " + data_dumm[key][i-1]["sentence_id"] + ". " + str(data[key][i-1]["speaker"]) +" : " + str(data[key][i-1]["sentence"]) + " \n "
            else:
                data_dumm_history[key][i]['history'] = "历史记录: \n " + data_dumm[key][i-3]["sentence_id"] + ". " + str(data[key][i-3]["speaker"]) +" : " + str(data[key][i-3]["sentence"]) + " \n " + data_dumm[key][i-2]["sentence_id"] + ". " + str(data[key][i-2]["speaker"]) +" : " + str(data[key][i-2]["sentence"]) + " \n " + data_dumm[key][i-1]["sentence_id"] + ". " + str(data[key][i-1]["speaker"]) +" : " + str(data[key][i-1]["sentence"]) + " \n "
            ## 句子整合
            data_dumm_history[key][i]["sentence"] = "当前对话： \n" + data_dumm[key][i]["sentence_id"] + ". " + data_dumm[key][i]["speaker"] + " : " + data_dumm[key][i]["sentence"]
            ## 添加标签
            data_dumm_history[key][i]["label"] = int(data_dumm[key][i]["dialogue_act"])
            data_list.append(data_dumm_history[key][i])

    for i in range(2):
        print(data_list[i+2])
        
    filename = f"./data/IMCS-DAC/{data_type}_list_data.json"

    # 使用json.dump()函数将列表写入文件
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False)

    print(f"数据已成功保存至 {filename}")