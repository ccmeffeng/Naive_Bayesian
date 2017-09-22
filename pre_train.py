import jieba
import random
import re
import numpy as np

class DataPrepare:
    def __init__(self, input_file, train_data_file, test_data_file, train_file_percentage):
        self.input_file = open(input_file,'r',encoding='utf-8').readlines()
        self.train_data_file = open(train_data_file,'w',encoding='utf-8')
        self.test_data_file = open(test_data_file,'w',encoding='utf-8')
        self.train_file_percentage = train_file_percentage
        self.unique_words = []
        # 每一个单词都使用一个数字类型的id表示，python索引的时候才会快一些
        self.word_ids = {}
    def __del__(self):
        self.train_data_file.close()
        self.test_data_file.close()
    def prepare(self):
        corpus = []
        stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
        stopwords = re.split(r'\n', stopwords)
        tag = None
        for sentence in self.input_file:
            if sentence == '\n':
                continue
            if re.search('000',sentence):
                tag = re.search('000\d+', sentence).group()
                continue
            seg_list = list(jieba.cut(sentence))
            # 去除停用词
            seg_list_2 = []
            for w in seg_list:
                if w not in stopwords and w != '\n':
                    seg_list_2.append(w)
            corpus.append([tag] + seg_list_2)

        for line in corpus:
            # 随即函数按照train_file_percentage指定的百分比来选择训练和测试数据
            if random.random() < self.train_file_percentage:
                output_file = self.train_data_file
            else:
                output_file = self.test_data_file
            # 读取文件获得词组
            output_file.write(line.pop(0) + ' ')
            for word in line:
                if word not in self.word_ids:
                    self.unique_words.append(word)
                    # 可以取Hash，这里为了简便期间，直接使用当前数组的长度（也是唯一的）
                    self.word_ids[word] = len(self.unique_words)
                output_file.write(str(self.word_ids[word]) + ' ')
            output_file.write('\n')
        trans = open('trans.model','w',encoding='utf-8')
        for key in self.word_ids:
            trans.write(key+' '+str(self.word_ids[key])+'\n')
        trans.close()



if __name__ == '__main__':
    dp = DataPrepare('classify_data.txt', 'train.txt', 'valid.txt', 1)
    dp.prepare()
