# -*- coding: utf-8 -*-
import json
import os

import requests
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import np_utils
import numpy as np
import sys
if sys.version.startswith("2."):
    reload(sys)
    sys.setdefaultencoding('utf8')

class RequestHandler():
    def __init__(self):
        self.models = self.get_models(path='/home/W2V/model/')
        self.models37 = self.get_models(path='/home/open/model37/')
        # self.models = self.get_models(path='model/')
        # self.models37 = self.get_models(path='model37/')
        self.key = 'Key.csv'
        self.tv =  'tv.txt'
        self.health = 'health.txt'
        self.radio = 'radio.txt'

    def getBatchResults(self,sentencesList):
        # 第一步：规则判断open1——————————————————————————————————————————————————————————————
        reader = open(self.key, 'r')
        labellist = []
        key = []
        for line in reader.readlines():
            temp = line.strip().split(',')
            labellist.append(temp[0])
            key.append(temp[1:])
        # print(len(labellist),len(key[0]))

        ResultsList = []
        for sentence in sentencesList:
            resultDict = {}
            for i in range(len(labellist)):
                flag = False
                for j in range(len(key[i])):
                    if key[i][j] in sentence['content']:
                        # print(key[i][j])
                        resultDict['id'] = sentence['id']
                        resultDict['result'] = labellist[i]
                        sentence['id']=None
                        ResultsList.append(resultDict)
                        flag = True
                        break
                if flag:
                    break
        newSentenceslistONE = []  # 保存规则之后剩余的句子
        for sentence in sentencesList:
            if sentence['id'] == None:
                pass
            else:
                newSentenceslistONE.append(sentence)

        # 第二步：规则判断open2电视台—找出37—————————————————————————————————————————————————————————————
        reader = open(self.tv, 'r')
        key37 = []
        for line in reader.readlines():
            if line.strip().__len__()>0:
                key37.append(line.strip())

        print(str(newSentenceslistONE))
        newSentenceslist37 = []
        newSentenceslistTWO = []
        for sentence in newSentenceslistONE:
            flag = False
            for keystr in key37:
                if keystr in sentence['content']:
                    flag = True
                    break
            if flag:
                newSentenceslist37.append(sentence)
            else:
                newSentenceslistTWO.append(sentence)

        label37 = ''
        if newSentenceslist37.__len__()>0:
            label37 = self.evaluate_model(self.sentenceslist2wv(newSentenceslist37))

        # 第二步：规则判断open2电视台—31分类中非37—————————————————————————————————————————————————————————————
        newSentenceslist37_ = []
        for i in range(label37.__len__()):
            if label37[i] in [ 'epg', 'tvchannel']:
                resultDict = {}
                resultDict['id'] = newSentenceslist37[i]['id']
                resultDict['result'] = label37[i]
                ResultsList.append(resultDict)
            else:
                newSentenceslist37_.append(newSentenceslist37[i])

        # 第二步：规则判断open2电视台—归入37—————————————————————————————————————————————————————————————
        if newSentenceslist37_.__len__()>0:
            label37 = self.evaluate_model37(self.sentenceslist2wv(newSentenceslist37_))
        i = 0
        for sentence in newSentenceslist37_:
            resultDict = {}
            resultDict['id'] = sentence['id']
            resultDict['result'] = label37[i]
            ResultsList.append(resultDict)
            i += 1

        # 第三步：规则判断open2疾病——————————————————————————————————————————————————————————————
        reader = open(self.health, 'r')
        keyHealth = []
        for line in reader.readlines():
            if line.strip().__len__() > 0:
                keyHealth.append(line.strip())

        newSentenceslistTHREE = []
        for sentence in newSentenceslistTWO:
            flag = False
            for keystr in keyHealth:
                if keystr in sentence['content']:
                    flag = True
                    break
            if flag:
                resultDict = {}
                resultDict['id'] = sentence['id']
                resultDict['result'] = 'health'
                ResultsList.append(resultDict)
            else:
                newSentenceslistTHREE.append(sentence)
        # 第四步：规则判断open2广播——————————————————————————————————————————————————————————————
        reader = open(self.radio, 'r')
        keyRadio = []
        for line in reader.readlines():
            if line.strip().__len__() > 0:
                keyRadio.append(line.strip())

        newSentenceslistFOUR = []
        keyRadio_ = ['听','频','播','台','FM']
        for sentence in newSentenceslistTHREE:
            flag = False
            for xkeystr in keyRadio_:
                if xkeystr in sentence['content']:
                    for keystr in keyRadio:
                        if keystr in sentence['content']:
                            flag = True
                            break
                    if flag:
                        break
            if flag:
                # print(sentence['id'],"radio")
                resultDict = {}
                resultDict['id'] = sentence['id']
                resultDict['result'] = 'radio'
                ResultsList.append(resultDict)
            else:
                # print(sentence['id'],"not radio")
                newSentenceslistFOUR.append(sentence)

        # 第五步：跑模型——————————————————————————————————————————————————————————————
        pre_label = ""
        if newSentenceslistFOUR.__len__() > 0:
            pre_label = self.evaluate_model(self.sentenceslist2wv(newSentenceslistFOUR))
        i = 0
        for sentence in newSentenceslistFOUR:
            resultDict = {}
            resultDict['id'] = sentence['id']
            resultDict['result'] = pre_label[i]
            ResultsList.append(resultDict)
            i += 1

        return ResultsList

    def sentenceslist2wv(self,Sentenceslist):
        url = "http://119.23.174.193:51242/smp2017_ecdt"
        parameter = {'sentencesList': Sentenceslist}
        headers = {'Content-type': 'application/json'}
        try:
            r = requests.post(url, data=json.dumps(
                parameter), headers=headers, timeout=4)
            if r.status_code == 200:
                data = r.json()
                resultsW2VList = data['resultsList']
                # for result in resultsW2VList:
                #     print(result['id'], result['result'])
            else:
                print("wrong,status_code: ", r.status_code)
        except Exception as e:
            print(Exception, ' : ', e)
        return resultsW2VList

    def get_models(self, path):
        '''

        :param path: 模型文件的目录
        :return: 目录下的模型文件列表
        '''
        filenames = os.listdir(path)
        models = list()
        max_val = 0
        for filename in filenames:
            path_file = path + filename  # 获取文件的相对路径
            model = load_model(path_file)  # 获取模型
            # 获取模型的测试准确率
            temp = filename.replace('.hdf5', '')
            temps = temp.split('val_acc_')
            val = float(temps[1])
            if val > max_val:
                models.insert(0, model)
                max_val = val
            else:
                models.append(model)
                # print val
        return models

    def predict_classes_integrated(self, models, resultsW2VList, nbclass):
        x_test = []
        for senetnces in resultsW2VList:
            x_test.append(senetnces['result'])
        x_test_len = x_test.__len__()

        y_predict_integrated = np.zeros((x_test_len, nbclass))
        # print(x_test)
        maxlen = 20  # cut texts after this number of words (among top max_features most common words)
        # 限定最大词数
        len_wv = 50
        # Memory 足够时用
        # print(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float64')
        x_test = x_test.reshape((x_test_len, maxlen, len_wv))
        # for model in models:
        #     y_predict = model.predict_classes(x_test, batch_size=128, verbose=0)
        #     temp = np_utils.to_categorical(y_predict, 31)
        #     y_predict_integrated += temp

        y_predict_val = models[0].predict_classes(x_test)  # 备份最大准确率的模型的分类结果
        temp=np_utils.to_categorical(y_predict_val,nbclass)
        y_predict_integrated+=temp
        for i in range(1,len(models)):
            y_predict = models[i].predict_classes(x_test, batch_size=128, verbose=0)
            temp = np_utils.to_categorical(y_predict, nbclass)
            y_predict_integrated += temp

        y_predict_copy = np.zeros(len(y_predict_integrated), dtype="uint8")
        for i in range(0, len(y_predict_integrated)):
            max_index = 0
            max_val_num = 0
            max_val = max(y_predict_integrated[i, :])
            for j in range(0, len(y_predict_integrated[i, :])):
                # if y_predict_integrated[i, j] > y_predict_integrated[i, max_index]:
                if y_predict_integrated[i, j] == max_val:
                    max_index = j
                    max_val_num += 1
            if max_val_num == 1:
                y_predict_copy[i] = max_index
            else:
                # print("some numbers of labels are equal!!")
                y_predict_copy[i] = y_predict_val[i]

        return y_predict_copy

    def evaluate_model(self, resultsW2VList):
        '''
        :param path_models: 模型文件的目录
        :return: 集成类别
        '''
        models = self.models

        label = ['chat', 'cookbook', 'video', 'epg', 'poetry', 'stock', 'train', 'tvchannel', 'map',
                 'weather', 'music', 'telephone', 'message', 'flight', 'translation', 'news', 'health', 'website',
                 'app', 'riddle', 'contacts', 'schedule', 'radio', 'lottery', 'cinemas', 'calc', 'email',
                 'match', 'bus', 'novel', 'datetime']

        predict_label = self.predict_classes_integrated(models, resultsW2VList, nbclass=31)

        final_label = []
        for i in range(len(predict_label)):
            final_label.append(label[predict_label[i]])

        return final_label

    def evaluate_model37(self, resultsW2VList):
        '''
        :param path_models: 模型文件的目录
        :return: 集成类别
        '''
        models = self.models37

        label = [ 'epg', 'tvchannel']

        predict_label = self.predict_classes_integrated(models, resultsW2VList, nbclass=2)

        final_label = []
        for i in range(len(predict_label)):
            final_label.append(label[predict_label[i]])

        return final_label



if __name__ == '__main__':
    rHandler = RequestHandler()

    sentencesList = [{'id': 0, 'content': '你知道小黄鸡吗'},
                     {'id': 1, 'content': '银耳莲子粥怎么做'},
                     {'id': 2, 'content': '播放2005年励志动漫'},
                     {'id': 3, 'content': '哪个频道有刘德华出演的电视剧'},
                     {'id': 4, 'content': '你背首唐诗给我听听啊'},
                     {'id': 5, 'content': '我想了解地产板块股票现在怎么样'},
                     {'id': 6, 'content': '查寻上海到北京的火车票'},
                     {'id': 7, 'content': '打开宁波新闻'},
                     {'id': 8, 'content': '我想到科大讯飞走高速路线'},
                     {'id': 9, 'content': '明天天气怎样啊'},
                     {'id': 10, 'content': '歌曲白天不懂夜的黑'},
                     {'id': 11, 'content': '拨打幺三九幺三八九九三九九'},
                     {'id': 12, 'content': '万古天帝'}
                     ]
    resultsList = rHandler.getBatchResults(sentencesList)
    print(np.shape(resultsList))
    print(resultsList)

