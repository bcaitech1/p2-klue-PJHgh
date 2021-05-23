#!/usr/bin/env python
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings(action='ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import copy
import json
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import ElectraForSequenceClassification, ElectraTokenizer, ElectraConfig

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


# In[2]:


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type, mode):
    if mode == 'train':
        label = []
        for i in dataset[0]:
            label.append(label_type[i])
        out_dataset = pd.DataFrame({'sentence':dataset[1],
                                    'label':label})
    elif mode == 'test':
        label = [100 for i in range(len(dataset[0]))]
        out_dataset = pd.DataFrame({'sentence':dataset[0],
                            'label':label})
    else: raise
    return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir, mode):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
        # load dataset
        dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
        # preprecessing dataset
        dataset = preprocessing_dataset(dataset, label_type, mode)
    return dataset


# In[3]:


# Dataset 구성.
class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels, tokenizer, threshold=0.1):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
        self.tokenizer = tokenizer
        self.threshold = threshold
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return self._random_enk(item['input_ids']), item['attention_mask'], item['labels']

    def __len__(self):
        return len(self.labels)
    
    def _random_enk(self, sent):
        unknown_ids = self.tokenizer.encode('<unk>')[1:-1]
#         unknown_ids = self.tokenizer.encode('<mask>')[1:-1]

        decoded_ids = self.tokenizer.decode(sent, skip_special_tokens=False)
        ent = list(decoded_ids.split('</s>'))[0]

        encoded_ent_list = list(set(self.tokenizer.encode(ent)+[1]))

        for i, token in enumerate(sent):
            if token in encoded_ent_list: continue
            elif self.threshold > random.random(): 
                sent[i] = unknown_ids[0]
#                 break
        return sent

def tokenized_dataset(dataset, tokenizer):
    concat_entity = mask_and_input_generation(dataset, tokenizer)

    tokenized_dataset = tokenizer(concat_entity,
                                  list(dataset['sentence']),
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=200,
                                  add_special_tokens=True)
    return tokenized_dataset

def mask_and_input_generation(dataset, tokenizer):
    token_e1_f = tokenizer.encode('<e1>')[2:-1]
    token_e1_b = tokenizer.encode('</e1>')[1:-2]

    token_e2_f = tokenizer.encode('<e2>')[2:-1]
    token_e2_b = tokenizer.encode('</e2>')[1:-2]

    token_bos = tokenizer.encode('<s>')[1:-1]
    token_eos = tokenizer.encode('</s>')[1:-1]
    token_sep = tokenizer.encode('RELATION')[1:-1]
    
    concat_entity = []
    e1_mask_list = []
    e2_mask_list = []
    for sent in dataset['sentence']:
        token_sent = tokenizer.encode(sent)
        for i in range(len(token_sent)-3):
            if token_e1_f == token_sent[i:i+3]:
                e1_f_idx = i+3
                break
        for i in range(len(token_sent)-4):
            if token_e1_b == token_sent[i:i+4]:
                e1_b_idx = i
                break
        for i in range(len(token_sent)-3):
            if token_e2_f == token_sent[i:i+3]:
                e2_f_idx = i+3
                break
        for i in range(len(token_sent)-4):
            if token_e2_b == token_sent[i:i+4]:
                e2_b_idx = i
                break

        e1 = token_sent[e1_f_idx:e1_b_idx]
        e2 = token_sent[e2_f_idx:e2_b_idx]
        
        temp = e1 + token_sep + e2
        concat_entity.append(tokenizer.decode(temp))
        
        e1_f_idx = 1e9
        e1_b_idx = 1e9
        e2_f_idx = 1e9
        e2_b_idx = 1e9
    return concat_entity

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes=42, smoothing=0.6, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
# In[7]:


class Trainer(object):
    def __init__(self,
                 tokenized_datasets,
                 labels,
                 tokenizer,
                 threshold,
                 model,
                 batch_size,
                 num_epochs,
                 weight_decay,
                 learning_rate,
                 accumulation_steps):
        
        self.input_ids = tokenized_datasets['input_ids']
        self.attention_mask = tokenized_datasets['attention_mask']

        index_list = list(range(len(self.input_ids)))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.kfolds = kf.split(index_list)
        
        self.labels = labels
        self.tokenizer = tokenizer
        self.threshold = threshold
        
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.accumulation_steps = accumulation_steps
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                                             {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.loss_fct = nn.CrossEntropyLoss()
#         self.loss_fct = LabelSmoothingLoss()
        self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
        
        # load test datset
        test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
        test_dataset, test_label = self.load_test_dataset(self.tokenizer)
        self.test_dataset = NLPDataset(test_dataset ,test_label, self.tokenizer, self.threshold)
        
    def train(self, save_path):
        since = time.time()
        
        initial_model_wts = copy.deepcopy(self.model.state_dict())
        for fold_counter ,(train_idx, valid_idx) in enumerate(self.kfolds):
            train_datasets = {'input_ids':Subset(self.input_ids, train_idx),
                              'attention_mask': Subset(self.attention_mask, train_idx)}
            valid_datasets = {'input_ids':Subset(self.input_ids, valid_idx),
                              'attention_mask': Subset(self.attention_mask, valid_idx)}

            train_labels = Subset(self.labels, train_idx)
            valid_labels = Subset(self.labels, valid_idx)

            train_datasets = NLPDataset(train_datasets, train_labels, self.tokenizer, threshold=self.threshold)
            valid_datasets = NLPDataset(valid_datasets, valid_labels, self.tokenizer, threshold=self.threshold)
        
            train_sampler = RandomSampler(train_datasets)
            train_dataloader = DataLoader(train_datasets,
                                          sampler=train_sampler,
                                          batch_size=self.batch_size,
                                          num_workers=4)

            valid_dataloader = DataLoader(valid_datasets,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=4)

            total_steps = len(train_dataloader) * self.num_epochs

            # -- logging
            log_dir = os.path.join(save_path, f'{fold_counter}_fold')
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            logger = SummaryWriter(log_dir=log_dir)

            # Train!
            print('='*100)
            print(f'{fold_counter} fold training start !!')
            best_loss = 1e9
            best_acc = 0.0
            global_step = 0
            
            self.model.load_state_dict(initial_model_wts)
            for epoch in range(self.num_epochs):
                start_time = time.time()
                print('='*100)
                print(f'Epoch {epoch}/{self.num_epochs-1}')
                for phase in ['train', 'valid']:
                    if phase == 'train':
                        epoch_iterator = tqdm(train_dataloader, desc=f"{phase} Iteration", leave=True)
                        self.model.train()  # Set model to training mode
                    else:
                        epoch_iterator = tqdm(valid_dataloader, desc=f"{phase} Iteration", leave=True)
                        self.model.eval()   # Set model to evaluate mode

                    running_loss, running_corrects, num_cnt = 0, 0, 0
                    self.optimizer.zero_grad()
                    for step, batch in enumerate(epoch_iterator):
                        with torch.set_grad_enabled(phase == 'train'):
                            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                            outputs = self.model(input_ids=batch[0],
                                                 attention_mask=batch[1],
                                                 return_dict=False)[0]
                            _, preds = torch.max(outputs, 1)

                            loss = self.loss_fct(outputs, batch[-1])
                            if phase == 'train':
                                loss.backward()
                                if (step+1) % self.accumulation_steps == 0:
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()

                                    logger.add_scalar(f"Train/loss", loss.item(), epoch*len(epoch_iterator) + step)
                                    logger.add_scalar(f"Train/accuracy", torch.sum(preds == batch[-1]).item()/len(batch[-1])*100, epoch*len(epoch_iterator) + step)

                            # statistics
                            running_loss += loss.item() * len(batch[-1])
                            running_corrects += torch.sum(preds.cpu() == batch[-1].cpu())
                            num_cnt += len(batch[-1])
                            global_step += 1

                    epoch_loss = float(running_loss / num_cnt)
                    epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)

                    if phase == 'valid':
                        logger.add_scalar(f"Val/loss", epoch_loss, epoch)
                        logger.add_scalar(f"Val/accuracy", epoch_acc, epoch)

                    print(f'fold-{fold_counter} global step-{global_step} | Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}')
                    if phase == 'valid' and epoch_loss < best_loss:
                        best_idx = epoch
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        print(f'\tfold : {fold_counter} ==> best model saved - {best_idx} / Loss: {best_loss:.4f}, Accuracy: {best_acc:.4f}')
                end_time = time.time() - start_time
                print(f'\t1 EPOCH Training complete in {end_time//60}m {end_time%60:.2f}s', end='\n\n')
            time_elapsed = time.time() - since
            print(f'{fold_counter} fold Training complete in {time_elapsed//60}m {time_elapsed%60:.2f}s')
            print(f'{fold_counter} fold Best valid Acc: {best_idx} - {best_acc:.4f}')

            # load best model weights
            self.model.load_state_dict(best_model_wts)
            print(f'\t>>> {fold_counter} fold Training complete !!')
#             torch.save(self.model.state_dict(), os.path.join(save_path, f'fold{fold_counter}_best_model.pt'))
#             print('model saved !!')
            
            print(f'{fold_counter} fold inference start !!')
            # predict answer
            logit_predictions = self.inference(self.model, self.test_dataset, self.device)
            # make csv file with predicted answer
            # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

            if fold_counter == 0:
                oof_soft_pred = logit_predictions
            else:
                oof_soft_pred += logit_predictions
            print(f'\t>>> {fold_counter} fold inference complete !!')
        return oof_soft_pred

    def inference(self, model, tokenized_sent, device):
        dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
        model.eval()
        
        logit_predictions = []
        iterator = tqdm(dataloader, desc=f"Inference Iteration")
        for i, batch in enumerate(iterator):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)  # GPU or CPU
                outputs = model(input_ids=batch[0],
                                attention_mask=batch[1],
                                return_dict=False)[0]
                logits = F.softmax(outputs, dim=1)
                logits[:,0] += 0.1
                logit_predictions.extend(logits.detach().cpu().numpy())
        return np.array(logit_predictions)

    def load_test_dataset(self, tokenizer):
        te_dataset_dir = '/opt/ml/RBERT-test.tsv'
        test_dataset = load_data(te_dataset_dir, 'test')
        test_label = test_dataset['label'].values
        # tokenizing dataset
        test_datasets = tokenized_dataset(test_dataset, tokenizer)
        input_ids = test_datasets['input_ids']
        attention_mask = test_datasets['attention_mask']

        test_datasets = {'input_ids':input_ids,
                         'attention_mask': attention_mask}
        return test_datasets, test_label
    
# In[8]:

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    seed_everything(seed=42)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu': raise
    
    config = AutoConfig.from_pretrained(args.MODEL_NAME)
    config.num_labels = 42
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(args.MODEL_NAME, config=config).to(device)
    
    # load dataset
    tr_dataset_dir = '/opt/ml/RBERT-train.tsv'
    dataset = load_data(tr_dataset_dir, 'train')
    labels = dataset['label'].values

    tokenized_datasets = tokenized_dataset(dataset, tokenizer)
    
    trainer = Trainer(tokenized_datasets=tokenized_datasets,
                      labels=labels,
                      tokenizer=tokenizer,
                      threshold=args.threshold,
                      model=model,
                      batch_size=32,
                      num_epochs=7,
                      weight_decay=0.001,
                      learning_rate=args.learning_rate,
                      accumulation_steps=1)
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    else: raise
    
    oof_soft_pred = trainer.train(args.save_path)
    
    save_dir = os.path.join(args.save_path, 'prediction')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    pred_answer = np.argmax(oof_soft_pred, axis=-1)
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--MODEL_NAME', type=str, default='monologg/koelectra-base-v3-discriminator')
    parser.add_argument('--add_unk_token', type=str2bool, default=False)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--dr_rate', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='./baseline')
    
    args = parser.parse_args()
    
    print(f'MODEL ::: {args.MODEL_NAME}')
    print(f'USE UNK TOKEN ::: {args.add_unk_token}')
    print(f'Validation Ratio ::: {args.val_ratio}')
    print(f'Learning Rate ::: {args.learning_rate}')
    print(f'DropOut Ratio ::: {args.dr_rate}')
    print(f'Threshold ::: {args.threshold}')
    print(f'SAVE PATH ::: {args.save_path}')
    
    main(args)
