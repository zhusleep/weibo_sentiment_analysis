import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizer, get_constant_schedule_with_warmup
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch import nn
import json
from apex import amp
from tqdm import tqdm as tqdm
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import KFold


usual_train = pd.read_excel('2020_SMP_raw_data/usual_train.xlsx')
virus_train = pd.read_excel('2020_SMP_raw_data/virus_train.xlsx')
usual_test = pd.read_excel('2020_SMP_raw_data/usual_eval.xlsx')
virus_test = pd.read_excel('2020_SMP_raw_data/virus_eval.xlsx')

usual_train['type'] = 'usual'
usual_test['type'] = 'usual'
virus_train['type'] = 'virus'
virus_test['type'] = 'virus'
print(usual_train.head())
print(virus_train.head())
data = usual_train.append(virus_train)
data['文本'] = data['文本'].astype('str')
# 打乱数据
data = data.reset_index(drop=True)
print(data.shape)
label_dict = {}
for label in data['情绪标签'].unique():
    label_dict[label] = len(label_dict)
print(label_dict)
data['情绪标签'] = data['情绪标签'].map(label_dict)


# 定义基本组件
tokenizer = BertTokenizer.from_pretrained("ernie")
config = BertConfig.from_json_file('ernie/config.json')
bert_path = 'ernie'
config.num_labels = len(label_dict)


class SentimentDataset(Dataset):

    def __init__(self, df, valid=False):
        self.raw_sentence = df['文本'].tolist()
        self.label = df['情绪标签'].tolist()
        self.type = df['type'].tolist()
        self._tokenizer = tokenizer
        self.max_len = 600
        self.sentence = self.sen_tokenize()

    def sen_tokenize(self):
        result = []
        for index, item in enumerate(self.raw_sentence):
            vector = self._tokenizer.encode(item[0:self.max_len])
            # # 添加标志位区分普通数据和疫情数据
            # if self.type[index]=='usual':
            #     vector.insert(1,1)
            # else:
            #     vector.insert(1,2)
            result.append(vector)
            # print(item)
            # print(self._tokenizer.decode(vector))
        return result

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return torch.tensor(self.sentence[idx]), self.label[idx]


def collate_fn(batch):
    token, label = zip(*batch)
    label = torch.tensor(label)
    token = pad_sequence(token, batch_first=True)
    return token,label


class SentimentModel(nn.Module):
    def __init__(self, config, bert_model):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.lstm = nn.LSTM(768,config.hidden_size,1,batch_first=True)
        # self.classify = nn.Sequential(
        #     # nn.BatchNorm1d(config.hidden_size),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=config.hidden_size, out_features=config.num_labels)
        # )

    def forward(
        self,
        input_ids=None,
        attention_mask=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[1]
        logits = self.classifier(sequence_output)
        return logits


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    lr = lr[0]
    return lr


batch_size = 4
lr = 3e-5
weight_decay = 0
adam_epsilon = 1e-8
n_epochs = 2
step = 1
warmup = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
kfold = KFold(n_splits=5, shuffle=True, random_state=2019)

r = 0
for train_index, test_index in kfold.split(np.zeros(len(data))):
    train = data.loc[train_index,:].reset_index()
    val = data.loc[test_index,:].reset_index()

    model = SentimentModel(config, bert_path)
    model.to(device)

    #  准确训练模型
    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW([
            {'params': model.bert.parameters(), 'lr': 2e-5}
        ], lr=1e-3)

    t_total = int(len(train)*n_epochs/batch_size)
    warmup_steps = int(t_total*warmup)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2', verbosity=0)

    report_each = 100
    loss_fn = nn.CrossEntropyLoss()
    for e in range(n_epochs):
        model.train()
        train_losses = []
        train_set = SentimentDataset(train)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_set = SentimentDataset(val, valid=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        for i, (token, label) in tqdm(enumerate(train_loader)):
            input_mask = (token > 0).to(device)
            token, label = token.to(device), label.to(device)
            outputs = model(input_ids=token, attention_mask=input_mask)
            loss = loss_fn(outputs, label)
            # if (i + 1) % step == 0:
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            # else:
            #     loss.backward()

            train_losses.append(loss.item())
            mean_loss = np.mean(train_losses[-report_each:])
            if i % 1500 == 0:
                print('loss: ', mean_loss)
            lr = get_learning_rate(optimizer)
        # validate
        model.eval()
        valid_losses = 0
        pred_set = []
        for i, (token, label) in tqdm(enumerate(valid_loader)):
            input_mask = (token > 0).to(device)
            token, label = token.to(device), label.to(device)
            with torch.no_grad():
                outputs = model(input_ids=token, attention_mask=input_mask)
            loss = loss_fn(outputs,label)
            valid_losses += loss.item()
            pred_set.append(outputs.cpu().numpy())

        # valid_loss = valid_loss / len(dev_X)
        pred_set = np.concatenate(pred_set, axis=0)
        label_set = val['情绪标签']
        top_class = np.argmax(pred_set, axis=1)
        equals = top_class == label_set
        accuracy = np.mean(equals)
        usual_acc = np.mean(equals[val['type']=='usual'])
        virus_acc = np.mean(equals[val['type']=='virus'])
        print('acc %f, usual acc %f, virus acc %f' % (accuracy,usual_acc,virus_acc))
        print('epoch %d, train loss　%f, val loss %f' % (r, sum(train_losses)/len(train_loader), valid_losses/len(valid_loader)))
    torch.save(model.state_dict(), 'model/model_%d.pth' % r)
    r += 1

# submit
usual_test['情绪标签'] = -1
virus_test['情绪标签'] = -1
usual_test_set = SentimentDataset(usual_test, valid=True)
usual_test_loader = DataLoader(usual_test_set, batch_size=batch_size,shuffle=False, collate_fn=collate_fn)
virus_test_set = SentimentDataset(virus_test, valid=True)
virus_test_loader = DataLoader(virus_test_set, batch_size=batch_size,shuffle=False, collate_fn=collate_fn)

id_label = {}
for key,value in label_dict.items():
    id_label[value] = key


def make_prediction(test_data, df):
    preds = None
    for r in range(5):
        model.load_state_dict(torch.load('model/model_%d.pth'%r))
        model.eval()
        pred_set = []
        for i, (token, label) in enumerate(test_data):
            input_mask = (token > 0).to(device)
            token, label = token.to(device), label.to(device)
            with torch.no_grad():
                outputs = model(input_ids=token, attention_mask=input_mask)
            pred_set.append(nn.Softmax(dim=1)(outputs).cpu().numpy())
        pred_set = np.concatenate(pred_set, axis=0)
        if preds is not None:
            preds += pred_set
        else:
            preds = pred_set

    top_class = np.argmax(preds, axis=1)
    result = []
    for index, id in enumerate(df['数据编号']):
        line = {}
        line['id'] = id
        line['label'] = id_label[top_class[index]]
        result.append(line)
    return result


usual_result = make_prediction(usual_test_loader, usual_test)
virus_result = make_prediction(virus_test_loader, virus_test)

with open('usual_result.txt', 'w', encoding='utf-8') as f:
    json.dump(usual_result, f)
with open('virus_result.txt', 'w', encoding='utf-8') as f:
    json.dump(virus_result, f)
# acc 0.710072, usual acc 0.700700, virus acc 0.740106
# acc 0.705052, usual acc 0.691643, virus acc 0.748021
# acc 0.703640, usual acc 0.697200, virus acc 0.724274
# train loss　0.709274, val loss 0.839431
# acc 0.701031, usual acc 0.692712, virus acc 0.727118