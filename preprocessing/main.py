# coding: utf-8
import joblib
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import sys
import gzip
import logging
import re
import cPickle as pickle
import gzip
import numpy as np
from operator import itemgetter
from collections import defaultdict
from gensim.models import Word2Vec
# from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold

reload(sys)
sys.setdefaultencoding("utf-8")


# ==========================================  SWDA 读取数据方案: 读取 W2V 拼接 保存 ==========================

smp_path = 'smp'
model_file = 'weibodata_vectorB.gem'
tags_file = 'smp11.tags'
data_file = 'smp_word_10G_pku.pretrain.pkl.gz'

word_pattern = re.compile(r'[a-z\']+')
except_words = ('and', 'of', 'to')
accept_words = ('i',)


def CorpusReader():
    # 符合SMP的 数据读取方式
    data = pd.read_csv(u'Chinese_data.csv',sep='\t', encoding='utf8', header=0)
    print('test data shape is :%s' % (str(data.shape)))
    data = data[['Tag', 'Sentence']]
    data['Words'] = [" ".join(jieba.cut(str(sentence))) for sentence in data['Sentence']]
    return data


# convert each utterance into a list of word vectors(presented as list),
# convert tag into it's number. return a list with element formed like
# ([word_vec1, word_vec2, ...], tag_no)
def process_data(model, tags):
    x = []
    y = []
    model_cache = {}
    non_modeled = set()
    corpus = CorpusReader()
    for index, utt in corpus.iterrows():

        wordlist = utt['Words'].split(' ')
        for word in wordlist:
            print word
            if word in model:
                if word not in model_cache:
                    model_cache[word] = model[word].tolist()
                    print
                    'dim: ' + str(len(model_cache[word]))  # ���ά�� ��ȡʱ��
            else:
                non_modeled.add(word)
        words = [model_cache[w] for w in wordlist if w in model_cache]
        tag = tags[utt['Tag']]  #tag 就是对应 tag_set 里的No.
        x.append(words)
        y.append(tag)
    print 'Complete. The following words are not converted: '
    print list(non_modeled)
    print len(list(non_modeled))   #输出没有被转为词向量的词个数 （删除处理）
    return (x, y)


def save_data(data, pickle_file):
    f = gzip.GzipFile(pickle_file, 'w')
    pickle.dump(data, f)
    f.close()


# load corpus and save number of tags into tags_file.
def preprocess_data():
    act_tags = defaultdict(lambda: 0)
    corpus = CorpusReader()
    for index, utt in corpus.iterrows():
        act_tags[utt['Tag']] += 1
    act_tags = act_tags.iteritems()
    act_tags = sorted(act_tags, key=itemgetter(1), reverse=True)
    f = open(tags_file, 'w')
    for k, v in act_tags:
        f.write('%s %d\n' % (k, v))
    f.close()  # save tag and its number accordingly
    return dict([(act_tags[i][0], i) for i in xrange(len(act_tags))])

def load_data(filename):
    '''
    读取数据，暂不设验证集
    :param dataset:
    :return:
    '''
    max_features = 2298
    TRAIN_SET = max_features
    f = gzip.open(filename, 'rb')
    # f=file(filename,'rb')
    data = pickle.load(f)
    f.close()
    data_x, data_y = data
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    test_x = data_x[TRAIN_SET:]
    test_y = data_y[TRAIN_SET:]
    return train_x, train_y, test_x, test_y


def cross_validation_split_for_smp_average(data_x,data_y,fold_num):
    '''
    function:将数据集平均划分为n_folds份,
    :param dataset: list    数据集
    :param fold_num: int    数据集划分的份数，即分几折
    :return: list   划分好的数据集列表，列表里没个元素都是一份数据集
    '''

    # 新建对象，防止划分数据集时对原始数据集产生影响
    data_x_copy=np.array(data_x)
    data_y_copy=np.array(data_y)

    # 划分k折
    folds_list=list()
    skf=StratifiedKFold(n_splits=fold_num)
    # i=0
    for train_index,test_index in skf.split(data_x_copy,data_y_copy):
        #根据下标获取数据
        train_x,test_x=np.array(data_x_copy[train_index]),np.array(data_x_copy[test_index])
        train_y,test_y=np.array(data_y_copy[train_index]),np.array(data_y_copy[test_index])
        # #设定文件名
        # filename_train_x = path_dir +filename+ '_train_x_' + str(i) + '.npy'
        # filename_train_y = path_dir + filename+'_train_y_' + str(i) + '.npy'
        # filename_test_x = path_dir + filename+'_test_x_' + str(i) + '.npy'
        # filename_test_y = path_dir + filename+'_test_y_' + str(i) + '.npy'
        # #保存文件
        # np.save(filename_train_x, train_x)
        # np.save(filename_train_y, train_y)
        # np.save(filename_test_x, test_x)
        # np.save(filename_test_y, test_y)
        # i+=1
        folds_list.append((train_x,train_y,test_x,test_y))
    return folds_list



def merget_data(array1,array2):
    pass
    array=np.concatenate((array1,array2),axis=0)
    if len(array)!=len(array1)+len(array2):
        print( 'error!!')
    return array

def main():
    print 'Preprocessing data ...'
    tags = preprocess_data()
    print 'Loading model ...'
    # model = KeyedVectors.load_word2vec_format(model_file, binary=True)
    # model = Word2Vec.load('weibodata_vectorB.gem')
    # model = Word2Vec.load(u'G:/ResearchOffice_540/10G语料/merge_10G_weibo_50.model')

    model=joblib.load('w2v.pkl')
    # model=joblib.load(u'E:/语料+训练工具/10G微博语料_pku/10G_retrofit_50_dim_not_biaozhun.pkl')

    print 'Reading and converting data from swda ...'
    data = process_data(model, tags)
    print 'Saving ...'
    save_data(data, data_file)

    x_train, y_train,x_develop,y_develop=load_data(data_file)
    fold_num=10
    folds_list_train=cross_validation_split_for_smp_average(x_train,y_train, fold_num)
    folds_list_develop = cross_validation_split_for_smp_average(x_develop, y_develop, fold_num)
    for i in range(fold_num):
        train_x1, train_y1, test_x1, test_y1=folds_list_train[i]
        train_x2, train_y2, test_x2, test_y2=folds_list_develop[i]
        train_x=merget_data(train_x1,train_x2)
        train_y=merget_data(train_y1,train_y2)
        test_x=merget_data(test_x1,test_x2)
        test_y=merget_data(test_y1,test_y2)

        np.save('10folds/smp_10merge_train_x_' + str(i)+'.npy', train_x)
        np.save('10folds/smp_10merge_train_y_' + str(i)+'.npy', train_y)
        np.save('10folds/smp_10merge_test_x_' + str(i)+'.npy', test_x)
        np.save('10folds/smp_10merge_test_y_' + str(i)+'.npy', test_y)



if __name__ == '__main__':
    main()
