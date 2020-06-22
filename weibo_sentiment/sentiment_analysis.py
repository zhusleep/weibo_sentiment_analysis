import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizer, get_constant_schedule_with_warmup
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch import nn
import json
from tqdm import tqdm as tqdm
import torch
from transformers import BertTokenizer, BertModel, BertConfig


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
data = data.sample(frac=1).reset_index(drop=True)
print(data.shape)
label_dict = {}
for label in data['情绪标签'].unique():
    label_dict[label] = len(label_dict)
print(label_dict)
data['情绪标签'] = data['情绪标签'].map(label_dict)


# 训练集、验证集划分
# data = data.loc[0:30000]
train_num = 30000
train = data.loc[0:train_num, :]
val = data.loc[train_num:, :]

# 定义基本组件
tokenizer = BertTokenizer.from_pretrained("rbt3")
config = BertConfig.from_json_file('rbt3/config.json')
config.num_labels = len(label_dict)


class SentimentDataset(Dataset):

    def __init__(self, df, valid=False):
        self.raw_sentence = df['文本'].tolist()
        self.label = df['情绪标签'].tolist()
        self.type = df['type'].tolist()
        self._tokenizer = tokenizer
        self.max_len = 320
        self.sentence = self.sen_tokenize()

    def sen_tokenize(self):
        result = []
        for index, item in enumerate(self.raw_sentence):
            vector = self._tokenizer.encode(item[0:self.max_len])
            # 添加标志位区分普通数据和疫情数据
            if self.type[index]=='usual':
                vector.insert(1,1)
            else:
                vector.insert(1,2)
            result.append(vector)
        return result

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return torch.LongTensor(self.sentence[idx]), self.label[idx]


def collate_fn(batch):
    token, label = zip(*batch)
    label = torch.LongTensor(label)
    token = pad_sequence(token, batch_first=True)
    return token,label


class SentimentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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


batch_size = 32
lr = 3e-5
weight_decay = 0
adam_epsilon = 1e-8
n_epochs = 4
step = 1
warmup = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = SentimentDataset(train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_set = SentimentDataset(val, valid=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, collate_fn=collate_fn)

model = SentimentModel(config)
model.to(device)

#  准确训练模型
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
t_total = int(len(train)*n_epochs/batch_size)
warmup_steps = int(t_total*warmup)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
report_each = 100
loss_fn = nn.CrossEntropyLoss()
for e in range(n_epochs):
    model.train()
    train_losses = []

    for i, (token, label) in tqdm(enumerate(train_loader)):
        input_mask = (token > 0).to(device)
        token, label = token.to(device), label.to(device)
        outputs = model(input_ids=token, attention_mask=input_mask)
        loss = loss_fn(outputs, label)
        if (i + 1) % step == 0:
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            scheduler.step()
        else:
            loss.backward()

        train_losses.append(loss.item())
        mean_loss = np.mean(train_losses[-report_each:])
        if i%100==0:
            print(mean_loss)
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
    print('acc', accuracy)
    print('train loss　%f, val loss %f' % (sum(train_losses)/len(train_loader), valid_losses/len(valid_loader)))
    # torch.save(model.state_dict(), '../model/deep_type_t_%d.pth' % round)

# submit
usual_test['情绪标签'] = -1
virus_test['情绪标签'] = -1
usual_test_set = SentimentDataset(usual_test, valid=True)
usual_test_loader = DataLoader(usual_test_set, batch_size=batch_size,shuffle=False, collate_fn=collate_fn)
virus_test_set = SentimentDataset(virus_test, valid=True)
virus_test_loader = DataLoader(virus_test_set, batch_size=batch_size,shuffle=False, collate_fn=collate_fn)


def make_prediction(test_data, df):
    pred_set = []
    for i, (token, label) in tqdm(enumerate(test_data)):
        input_mask = (token > 0).to(device)
        token, label = token.to(device), label.to(device)
        with torch.no_grad():
            outputs = model(input_ids=token, attention_mask=input_mask)
        pred_set.append(outputs.cpu().numpy())
    pred_set = np.concatenate(pred_set, axis=0)
    top_class = np.argmax(pred_set, axis=1)
    result = {}
    for index, id in enumerate(df['数据编号']):
        result[str(id)] = str(top_class[index])
    return result


usual_result = make_prediction(usual_test_loader, usual_test)
virus_result = make_prediction(virus_test_loader, virus_test)

with open('usual_result.json', 'w', encoding='utf-8') as f:
    json.dump(usual_result, f)
with open('virus_result.json', 'w', encoding='utf-8') as f:
    json.dump(virus_result, f)
