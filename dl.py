import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import numpy as np
from ml import *

def dataloader(content):
    X_c = []
    y_c = []
    X_e = []
    y_e = []
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
                        if information[0] == 'ham':
                            y_c.append(1)
                        else :
                            y_c.append(0)
                    else:
                        content = remove(content)
                        content = textParse(content)
                        X_e.append(content)
                        if information[0] == 'ham':
                            y_e.append(1)
                        else :
                            y_e.append(0)
            except:
                continue
    return X_c,y_c,X_e,y_e

class FCNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(FCNet, self).__init__()
            self.fc_net = nn.Sequential(
                nn.Linear(input_size,hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(hidden_size,512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(512,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256,output_size),
                )
        
        def forward(self,x):
            return self.fc_net(x)
class Email(Dataset):
    def __init__(self,X,y):
        self.features = torch.tensor(torch.from_numpy(np.array(X)),dtype=torch.float32)
        self.labels = torch.from_numpy(np.array(y))
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx]

def trainval(input_size,X_train,y_train,X_test,y_test):
    #--------------------train--------------------
    #随机种子
    SEED = 0
    np.random.seed(SEED) 
    torch.manual_seed(SEED) 
    torch.cuda.manual_seed(SEED) 

    #配置参数
    hidden_size = 1024
    output_size = 2
    learning_rate = 0.001
    batch_num = 100
    epoch_num = 50

    #配置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #配置损失函数
    criterion = nn.CrossEntropyLoss()

    #配置模型
    model = FCNet(input_size,hidden_size,output_size).to(device)
    model.train()

    #配置优化器
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)

    #数据预处理
    data_train = Email(X_train,y_train)
    data_test = Email(X_test,y_test)
    datalo_train = DataLoader(data_train,batch_size=batch_num)
    datalo_test = DataLoader(data_test,batch_size=1)
    #网络训练
    for epoch in range(epoch_num):
        for i,(features,label) in enumerate(datalo_train):
            features = features.to(device)
            label_tackled = Variable(label).to(device)
            out = model(features)
            loss = criterion(out,label_tackled)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1)%2 == 0 :
            print(f'epoch:{epoch+1}/{epoch_num},loss:{loss.item():.6f}')
        
    #--------------------test--------------------
    model.eval()
    with torch.no_grad():
        #参数设置
        correct_num = 0
        total = 0
        #测试集设置

        for i,(features,label) in enumerate(datalo_test):
            features = features.to(device)
            out = model(features).to(device)
            sigmoid_pred = nn.Sigmoid()
            out = sigmoid_pred(out)
            out_cpu = out.to('cpu')
            prediction = torch.max(out_cpu,dim=1)
            if prediction[1].numpy() == label.numpy():
                correct_num += 1
            total += 1
        acc = correct_num / total
    #--------------------test--------------------
            
    #模型保存
    torch.save(model.state_dict(), f'model_{input_size}.pth')

    return acc

if __name__ == '__main__':
    t1 = time.time()
    X_c,y_c,X_e,y_e = dataloader('dataset/index_test.txt')
    print('----------中文数据集测试----------')
    num_chinese_sample = len(y_c)
    chinese_vocabulary = createVocaList(X_c)
    input_size_c = len(chinese_vocabulary)
    print('词汇表生成完毕。')
    X_c = wordbag_extractor(chinese_vocabulary,X_c)
    print('词袋特征提取完毕完毕。')
    X_c_train,X_c_test,y_c_train,y_c_test = train_test_split(X_c,y_c,train_size=0.7,random_state=0)
    print('数据集划分完毕。')
    acc_of_c = trainval(input_size_c,X_c_train,y_c_train,X_c_test,y_c_test)
    print('----------英文数据集测试----------')
    num_english_sample = len(y_e)
    english_vocabulary = createVocaList(X_e)
    input_size_e = len(english_vocabulary)
    print('词汇表生成完毕。')
    X_e = wordbag_extractor(english_vocabulary,X_e)
    print('词袋特征提取完毕完毕。')
    X_e_train,X_e_test,y_e_train,y_e_test = train_test_split(X_e,y_e,train_size=0.7,random_state=0)
    print('数据集划分完毕。')
    acc_of_e = trainval(input_size_e,X_e_train,y_e_train,X_e_test,y_e_test)

    acc = (acc_of_c * num_chinese_sample + acc_of_e * num_english_sample) / (num_chinese_sample + num_english_sample)

    print(acc)