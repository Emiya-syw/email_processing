import re
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB,ComplementNB,BernoulliNB,CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
import time
from nltk import FreqDist
import tqdm
import argparse
import sys
import scipy.sparse



def get_argument():
    parser = argparse.ArgumentParser(description='choose a feature extractor')
    parser.add_argument('--extractor',type=str,default='wordbag')
    parser.add_argument('--classifier',type=str,default='MuB')
    args = parser.parse_args()
    return args

def get_content(content):
    sta = re.search('<TEXT>',content)
    end = re.search('</TEXT>',content)
    content = content[sta.end()+1:end.start()]
    return content
# 判断是否为中文邮件
def is_chinese(content):
    content_enco = content.encode('utf-8')
    if len(content_enco) > len(content):
        return True
    return False
# 去除无用符号（中文）
def find_chinese (content):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese_txt = re.sub(pattern,'',content)
    return chinese_txt
# 中文分词
def chinese_textParse(content):
    content = list(jieba.cut(content))
    return [word for word in content if len(word)>=2]
# 去除停用词
def seg_sentence(content):
    with open('stopwords.txt','r') as f: 
        stopwords = f.readlines()
    seg_txt = [ w for w in content if w not in stopwords]
    return seg_txt
# 英文分词
def textParse(content):
    listOfTokens = content.split()
    return [tok.lower() for tok in listOfTokens if len(tok)>2 and len(tok)<15]
# 去除无用符号（英文）
def remove(content):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', content)
# 创建词汇表
def createVocaList(dataset):
    vocabSet = set({})
    for document in dataset:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 单样本词频统计,不考虑未出现词汇
def wordbag_extractor(vocabList,dataset):
    num_features = len(vocabList)
    num_samples = len(dataset)
    features = np.zeros((num_samples,num_features))
    for idx,inputSet in tqdm.tqdm(enumerate(dataset)):
        freq = FreqDist(inputSet)
        for key in freq.keys():
            features[idx,vocabList.index(key)] = freq[key]
    return features

def tfidf_extractor(count_matrix):
    tf_transformer = TfidfTransformer()
    tfdif_matrix = tf_transformer.fit_transform(count_matrix)
    return tfdif_matrix

def dataloader(content):
    X_c = []
    y_c = []
    X_e = []
    y_e = []
    count_ch = 0
    count_cs = 0
    count_eh = 0
    count_es = 0
    with open(content,'r',encoding='utf-8') as f: 
        path_to_samples = f.readlines()[1:]
        for sample in path_to_samples:
            if sample != '\n':
                information = sample.split()
                path = information[1][2:]
                path = './dataset' + path
            else:
                continue
            try:
                with open(path,'r',encoding='gbk') as f2:
                    content = f2.read()
                    content = get_content(content)
                    if is_chinese(content):
                        content = find_chinese(content)
                        content = chinese_textParse(content)
                        content = seg_sentence(content)
                        X_c.append(content)
                        y_c.append(information[0])
                        if information[0] == 'spam':
                            count_cs += 1
                        else:
                            count_ch += 1
                    else:
                        content = remove(content)
                        content = textParse(content)
                        X_e.append(content)
                        y_e.append(information[0])
                        if information[0] == 'spam':
                            count_es += 1
                        else:
                            count_eh += 1
            except:
                continue
    print(f'Chinese_Spam={count_cs},Chinese_Ham={count_ch},English_Spam={count_es},English_Ham={count_eh},')
    return X_c,y_c,X_e,y_e

if __name__ == '__main__':
    args = get_argument()

    t1 = time.time()
    X_c,y_c,X_e,y_e = dataloader('dataset/index_test.txt')
    
    

    print('----------中文数据集测试----------')
    num_chinese_sample = len(y_c)
    chinese_vocabulary = createVocaList(X_c)
    print('词汇表生成完毕。')

    if args.extractor == 'wordbag':
        X_c = wordbag_extractor(chinese_vocabulary,X_c)
        X_c = scipy.sparse.csr_matrix(X_c)
        print('词袋特征提取完毕完毕。')
    elif args.extractor == 'tfidf':
        X_c = wordbag_extractor(chinese_vocabulary,X_c)
        X_c = tfidf_extractor(X_c)
        print('tfidf特征提取完毕')
    else: 
        print('请选择一种特征提取方式！')
        sys.exit(0)

    X_c_train,X_c_test,y_c_train,y_c_test = train_test_split(X_c,y_c,train_size=0.7,random_state=0)
    print(type(X_c_train))
    print('数据集划分完毕。')

    if args.classifier == 'MuB':
        classifier_c = MultinomialNB()
    elif args.classifier == 'GuB':
        X_c_train = X_c_train.toarray()
        X_c_test = X_c_test.toarray()
        classifier_c = GaussianNB()
    elif args.classifier == 'CoB':
        classifier_c = ComplementNB()
    elif args.classifier == 'BeB':
        classifier_c = BernoulliNB()
    elif args.classifier == 'CaB':
        X_c_train = X_c_train.toarray()
        X_c_test = X_c_test.toarray()
        classifier_c = CategoricalNB()
    elif args.classifier == 'SVM':
        classifier_c = svm.LinearSVC()
    elif args.classifier == 'KNN':
        classifier_c = KNeighborsClassifier(n_neighbors=3)
    else:
        print('请选择一种分类方式')
        sys.exit()

    print('训练中......')
    classifier_c.fit(X_c_train,y_c_train)
    print('测试中......')
    acc_of_c = classifier_c.score(X_c_test,y_c_test)
    pred_c = classifier_c.predict(X_c_test)
    pr_c = precision_score(y_c_test,pred_c,pos_label='ham')
    re_c = recall_score(y_c_test,pred_c,pos_label='ham')
    f1_c = f1_score(y_c_test,pred_c,pos_label='ham')

    print(f'正确率为：{acc_of_c:.2f}  precision：{pr_c:.2f}  recall：{re_c:.2f}  f1score：{f1_c:.2f}')


    print('----------英文数据集测试----------')
    num_english_sample = len(y_e)
    english_vocabulary = createVocaList(X_e)
    print('词汇表生成完毕。')

    if args.extractor == 'wordbag':
        X_e = wordbag_extractor(english_vocabulary,X_e)
        X_e = scipy.sparse.csr_matrix(X_e)
        print('词袋特征提取完毕完毕。')
    elif args.extractor == 'tfidf':
        X_e = wordbag_extractor(english_vocabulary,X_e)
        X_e = tfidf_extractor(X_e)
        print('tfidf特征提取完毕')
    else: 
        print('请选择一种特征提取方式！')
        sys.exit(0)

    X_e_train,X_e_test,y_e_train,y_e_test = train_test_split(X_e,y_e,train_size=0.7,random_state=0)
    print('数据集划分完毕。')

    if args.classifier == 'MuB':
        classifier_e = MultinomialNB()
    elif args.classifier == 'GuB':
        X_e_train = X_e_train.toarray()
        X_e_test = X_e_test.toarray()
        classifier_e = GaussianNB()
    elif args.classifier == 'CoB':
        classifier_e = ComplementNB()
    elif args.classifier == 'BeB':
        classifier_e = BernoulliNB()
        X_e_train = X_e_train.toarray()
        X_e_test = X_e_test.toarray()
    elif args.classifier == 'CaB':
        classifier_e = CategoricalNB()
    elif args.classifier == 'SVM':
        classifier_e = svm.LinearSVC()
    elif args.classifier == 'KNN':
        classifier_e = KNeighborsClassifier(n_neighbors=10)
    else:
        print('请选择一种分类方式')
        sys.exit()

    print('训练中......')
    classifier_e.fit(X_e_train,y_e_train)
    print('测试中......')
    acc_of_e = classifier_e.score(X_e_test,y_e_test)
    pred_e = classifier_e.predict(X_e_test)
    pr_e = precision_score(y_e_test,pred_e,pos_label='ham')
    re_e = recall_score(y_e_test,pred_e,pos_label='ham')
    f1_e = f1_score(y_e_test,pred_e,pos_label='ham')
    
    print(f'正确率为：{acc_of_e:.2f}  precision：{pr_e:.2f}  recall：{re_e:.2f}  f1score：{f1_e:.2f}')

    acc = (acc_of_c * num_chinese_sample + acc_of_e * num_english_sample) / (num_chinese_sample + num_english_sample)
    gt = y_c_test + y_e_test
    pred = np.concatenate([pred_c,pred_e],axis=0)
    pr = precision_score(gt,pred,pos_label='ham')
    re = recall_score(gt,pred,pos_label='ham')
    f1 = f1_score(gt,pred,pos_label='ham')
    print(f'total accuracy = {acc:.2f}  precision = {pr:.2f}  recall = {re:.2f}  total f1score = {f1:.2f}')

    t2 = time.time()
    duration = t2 - t1
    print(f'实验结束，耗时{duration:.2f}s。')








