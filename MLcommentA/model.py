import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# 原始数据处理：读取新文件，加入字典和存储表格（0-neg,1-pos)
# 原始数据里有些条目无法读取，目前不能保证好评和差评数量是一致的
# 包含中英文数据集
d = {'contents':[],'labels':[]}
table = pd.DataFrame(columns=['contents','labels'])
with open(file='comments.txt', mode='rt', encoding='gbk') as f:
    for line in f:
        if line:
            if "neg" in line:
                a = line.strip().replace("--neg","")
                d['contents'].append(a)
                d['labels'].append(0)
                table.loc[len(table)]=[a,0]
            elif "pos" in line:
                a = line.strip().replace("--pos","")
                d['contents'].append(a)
                d['labels'].append(1)
                table.loc[len(table)]=[a,1]

with open(file='comments1.txt', mode='rt', encoding='gbk') as f:
    for line in f:
        if line:
            if "neg" in line:
                a = line.strip().replace("--neg","")
                d['contents'].append(a)
                d['labels'].append(0)
                table.loc[len(table)]=[a,0]
            elif "pos" in line:
                a = line.strip().replace("--pos","")
                d['contents'].append(a)
                d['labels'].append(1)
                table.loc[len(table)]=[a,1]

# 分词
X0 = d["contents"]
words = []
for i in range(len(X0)):
    word = jieba.lcut(X0[i])
    result = ' '.join(word)
    words.append(result)

# 文本向量化
vect = CountVectorizer()
X = vect.fit_transform(words).toarray()
y = d['labels']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 选定模型
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X=X_train, y=y_train)

# 保存模型
import joblib
joblib.dump(clf, "clf.pkl")
joblib.dump(vect, "vect.pkl")