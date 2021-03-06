class NavieBayes(object):
    """朴素贝叶斯模型"""
    def __init__(self,train_data_file,model_file):
        self.train_data_file = open(train_data_file,'r',encoding='utf-8')
        self.model_file = open(model_file,'w',encoding='utf-8')
        # 存储每一种类型出现的次数
        self.class_count = {}
        # 存储每一种类型下各个单词出现的次数
        self.class_word_count = {}
        # 唯一单词总数
        self.unique_words = {}
        # ~~~~~~~~~~ NavieBayes参数 ~~~~~~~~~~~~#
        # 每个类别的先验概率
        self.class_probabilities = {}
        # 拉普拉斯平滑，防止概率为0的情况出现
        self.laplace_smooth = 0.1
        # 模型训练结果集
        self.class_word_prob_matrix = {}
        # 当某个单词在某类别下不存在时，默认的概率（拉普拉斯平滑后）
        self.class_default_prob = {}
    def __del__(self):
        self.train_data_file.close()
        self.model_file.close()
    def loadData(self):
        words = self.train_data_file.readline().strip().split()
        while len(words) > 0:
            category = words.pop(0)
            if category not in self.class_count:
                self.class_count[category] = 0
                self.class_word_count[category] = {}
                self.class_word_prob_matrix[category] = {}
            self.class_count[category] += 1
            for word in words[1:]:
                word_id = int(word)
                if word_id not in self.unique_words:
                    self.unique_words[word_id] = 1
                if word_id not in self.class_word_count[category]:
                    self.class_word_count[category][word_id] = 1
                else:
                    self.class_word_count[category][word_id] += 1
            words = self.train_data_file.readline().strip().split()
    def computeModel(self):
        # 计算P(Yi)
        line_count = 0
        for count in self.class_count.values():
            line_count += count
        for class_id in self.class_count.keys():
            self.class_probabilities[class_id] = float(self.class_count[class_id]) / line_count
        # 计算P(X|Yi)  <===>  计算所有 P(Xi|Yi)的积  <===>  计算所有 Log(P(Xi|Yi)) 的和
        for class_id in self.class_word_count.keys():
            # 当前类别下所有单词的总数
            sum = 0.0
            for word_id in self.class_word_count[class_id].keys():
                sum += self.class_word_count[class_id][word_id]
            count_Yi = float(sum + len(self.unique_words)*self.laplace_smooth)
            # 计算单个单词在某类别下的概率，存储在结果矩阵中，所有当前类别没有的单词赋予默认概率(即使用拉普拉斯平滑)
            for word_id in self.class_word_count[class_id].keys():
                self.class_word_prob_matrix[class_id][word_id] = \
                    float(self.class_word_count[class_id][word_id]+self.laplace_smooth) / count_Yi
            self.class_default_prob[class_id] = float(self.laplace_smooth) / count_Yi
            print('%s matrix finished, length = %d' %(class_id, len(self.class_word_prob_matrix[class_id])))
        return
    def saveModel(self):
        # 把每个分类的先验概率写入文件
        for class_id in self.class_probabilities.keys():
            self.model_file.write(class_id)
            self.model_file.write(' ')
            self.model_file.write(str(self.class_probabilities[class_id]))
            self.model_file.write(' ')
            self.model_file.write(str(self.class_default_prob[class_id]))
            self.model_file.write('#')
        self.model_file.write('\n')
        # 把每个单词在当前类别的概率写入文件
        for class_id in self.class_word_prob_matrix.keys():
            self.model_file.write(class_id + ' ')
            for word_id in self.class_word_prob_matrix[class_id].keys():
                self.model_file.write(str(word_id) + ' '\
                     + str(self.class_word_prob_matrix[class_id][word_id]))
                self.model_file.write(' ')
            self.model_file.write('\n')
        return
    def train(self):
        self.loadData()
        self.computeModel()
        self.saveModel()
if __name__ == '__main__':
    nb = NavieBayes('train.txt','result.model')
    nb.train()
