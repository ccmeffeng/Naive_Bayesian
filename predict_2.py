import math
import re
import jieba

class NaiveBayesPredict(object):
    """使用训练好的模型进行预测"""
    def __init__(self, test_data_file, model_data_file):
        self.test_data = test_data_file
        self.model_data_file = open(model_data_file,'r',encoding='utf-8')
        # 每个类别的先验概率
        self.class_probabilities = {}
        # 拉普拉斯平滑，防止概率为0的情况出现
        self.laplace_smooth = 0.1
        # 模型训练结果集
        self.class_word_prob_matrix = {}
        # 当某个单词在某类别下不存在时，默认的概率（拉普拉斯平滑后）
        self.class_default_prob = {}
        # 所有单词
        self.unique_words = {}
        # 实际的新闻分类
        # self.real_classes = []
        # 预测的新闻分类
        self.predict_classes = []
    def __del__(self):
        self.model_data_file.close()
    def loadModel(self):
        # 从模型文件的第一行读取类别的先验概率
        class_probs = self.model_data_file.readline().split('#')
        for cls in class_probs:
            arr = cls.split()
            if len(arr) == 3:
                self.class_probabilities[arr[0]] = float(arr[1])
                self.class_default_prob[arr[0]] = float(arr[2])
        # 从模型文件读取单词在每个类别下的概率
        line = self.model_data_file.readline().strip()
        while len(line) > 0:
            arr = line.split()
            assert(len(arr) % 2 == 1)
            assert(arr[0] in self.class_probabilities)
            self.class_word_prob_matrix[arr[0]] = {}
            i = 1
            while i < len(arr):
                word_id = int(arr[i])
                probability = float(arr[i+1])
                if word_id not in self.unique_words:
                    self.unique_words[word_id] = 1
                self.class_word_prob_matrix[arr[0]][word_id] = probability
                i += 2
            line = self.model_data_file.readline().strip()
        print('%d classes loaded! %d words!' %(len(self.class_probabilities), len(self.unique_words)))
    def prepare(self):
        stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
        stopwords = re.split(r'\n', stopwords)
        sentence = open(self.test_data,'r',encoding='utf-8').read()
        assert sentence != '\n'
        seg_list = list(jieba.cut(sentence))
        # 去除停用词
        seg_list_2 = []
        for w in seg_list:
            if w not in stopwords:
                seg_list_2.append(w)
        return seg_list_2
    def calculate(self):
        # 读取测试数据集
        trans = {}
        for group in open('trans.model','r',encoding='utf-8').readlines():
            lst = group.split()
            assert len(lst) == 2
            trans[lst[0]] = int(lst[1])
        line = self.prepare()
        class_score = {}
        for key in self.class_probabilities.keys():
            class_score[key] = math.log(self.class_probabilities[key])
        for word_name in line:
            if word_name not in trans:
                continue
            word_id = trans[word_name]
            if word_id not in self.unique_words:
                continue
            for class_id in self.class_probabilities.keys():
                if word_id not in self.class_word_prob_matrix[class_id]:
                    class_score[class_id] += math.log(self.class_default_prob[class_id])
                else:
                    class_score[class_id] += math.log(self.class_word_prob_matrix[class_id][word_id])
        # 对于当前新闻，所属的概率最高的分类
        max_class_score = max(class_score.values())
        for key in class_score.keys():
            if class_score[key] == max_class_score:
                return key
        return None

    def predict(self):
        self.loadModel()
        return self.calculate()

if __name__ == '__main__':
    nbp = NaiveBayesPredict('valid.txt', 'result.model')
    print(nbp.predict())
