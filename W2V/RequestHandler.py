# -*- coding: utf-8 -*-
import jieba
import numpy as np
import sequence
import sys
if sys.version.startswith("2."):
    reload(sys)
    sys.setdefaultencoding('utf8')

import joblib


class RequestHandler():
    def __init__(self):
        self.model = joblib.load(u'w2v.pkl')
        pass

    def getResult(self, sentence):
        """1. 把句子转成向量

        Args:
            sentence: A string of sentence.

        Returns:
            句子向量
        """
        print(sentence)
        model_cache = {}
        non_modeled = set()
        wordlist = ' '.join(jieba.cut(str(sentence)))

        for word in wordlist.split(' '):
            # print(word)
            if word in self.model:
                if word not in model_cache:
                    model_cache[word] = self.model[word].tolist()
                    # print('dim: ' + str(len(model_cache[word])))

            else:
                non_modeled.add(word)
        # print(model_cache)

        words = [model_cache[w] for w in wordlist.split(' ') if w in model_cache]
        # print('words',words)
        return words

    def getBatchResults(self, sentencesList):
        """2. You can also complete the classification in this function,
                if you want to classify the sentences in batch.

        Args:
            sentencesList: A List of Dictionaries of ids and sentences,
                like:
                [{'id':331, 'content':'帮我打电话给张三' }, 
                 {'id':332, 'content':'帮我订一张机票!' },
                 ... ]

        Returns:
            resultsList: A List of Dictionaries of ids and results.
                The order of the list must be the same as the input list,
                like:
                [{'id':331, 'result':'telephone' }, 
                 {'id':332, 'result':'flight' },
                 ... ]
        """
        resultsList = []
        for sentence in sentencesList:
            resultDict = {}
            resultDict['id'] = sentence['id']
            resultDict['result'] = self.getResult(sentence['content'])
            resultsList.append(resultDict)
        return resultsList
        # senetncesVectorList = []
        # for sentence in sentencesList:
        #     senetncesVectorList.append(self.getResult(sentence['content']))
        # return senetncesVectorList


if __name__ == '__main__':
    rHandler = RequestHandler()

    sentencesList = [{'id': 0, 'content': '你知道小黄鸡吗'},
                     # {'id': 1, 'content': '银耳莲子粥怎么做'},
                     # {'id': 2, 'content': '播放2005年励志动漫'},
                     # {'id': 3, 'content': '哪个频道有刘德华出演的电视剧'},
                     # {'id': 4, 'content': '你背首唐诗给我听听啊'},
                     # {'id': 5, 'content': '我想了解地产板块股票现在怎么样'},
                     # {'id': 6, 'content': '查寻上海到北京的火车票'},
                     # {'id': 7, 'content': '打开浦口生活资讯频道'},
                     # {'id': 8, 'content': '我想到科大讯飞走高速路线'},
                     # {'id': 9, 'content': '明天天气怎样啊'},
                     # {'id': 10, 'content': '歌曲白天不懂夜的黑'},
                     {'id': 11, 'content': '拨打幺三九幺三八九九三九九'}
                     ]
    resultsList = rHandler.getBatchResults(sentencesList)
    print(np.shape(resultsList))
    print(resultsList)
    resultsList=sequence.pad_sequences(resultsList,maxlen=20,dtype='float64')
    resultsList=resultsList.reshape((len(resultsList),20,50))
    print(resultsList.shape)
