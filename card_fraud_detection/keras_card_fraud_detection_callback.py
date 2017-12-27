import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import collections

'''
카드 오용 탐지 keras 버전
url -> https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow
'''

# csv 파일 읽기
df = pd.read_csv("creditcard.csv")

# Create a new feature for normal (non-fraudulent) transactions.
# Normal 컬럼 생성 Class가 0 이면 1, Class가 1이면 0
df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0

# Rename 'Class' to 'Fraud'
# Class -> Fraud 로 명 변환
df = df.rename(columns = {'Class' : 'Fraud'})

# 492 fraudulent transactions, 284,315 normal transactions.
# 0.172% of transactions were fraud.
print(df.Normal.value_counts())
print()
print(df.Fraud.value_counts())

# max column 수 설정
pd.set_option('display.max_columns',  101)
# print(df.head())

# Create dataframes of only Fraud and Normal transactions.
Fraud = df[df.Fraud == 1]
Normal = df[df.Normal == 1]
print('Fraud  : ', len(Fraud))
print('Normal : ', len(Normal))

# Set X_train equal to 80% of the fraudulent transactions.
FraudSample  = Fraud.sample(frac=0.8)
NormalSample = Normal.sample(frac=0.8)
count_Frauds = len(FraudSample)

# Add 80% of the normal transactions to X_train.
for_train = pd.concat([FraudSample, NormalSample], axis=0)

# X_test contains all the transaction not in X_train. 20%
for_test = df.loc[~df.index.isin(for_train.index)]

print('len(for_train)  : ',len(for_train))
print('len(for_test)   : ',len(for_test))

#Shuffle the dataframes so that the training is done in a random order.
for_train = for_train.sample(frac=1).reset_index(drop=True)
for_test = for_test.sample(frac=1).reset_index(drop=True)

# Add our target features to y_train and y_test.
X_train = for_train.drop(['Fraud', 'Normal'], axis = 1)
# Drop target features from X_train and X_test.
# Fraud, Normal 컬럼 drop
y_train = for_train[['Fraud', 'Normal']]

# Add our target features to y_train and y_test.
X_test = for_test.drop(['Fraud', 'Normal'], axis = 1)
# Drop target features from X_train and X_test.
# Fraud, Normal 컬럼 drop
y_test = for_test[['Fraud', 'Normal']]

#Check to ensure all of the training/testing dataframes are of the correct length
print('len(X_train) : ',len(X_train))
print('len(y_train) : ',len(y_train))
print('len(X_test)  : ',len(X_test))
print('len(y_test)  : ',len(y_test))

# In [26]
'''
Due to the imbalance in the data, ratio will act as an equal weighting system for our model. 
By dividing the number of transactions by those that are fraudulent, ratio will equal the value that when multiplied
by the number of fraudulent transactions will equal the number of normal transaction. 
Simply put: # of fraud * ratio = # of normal
'''

ratio = len(X_train) / count_Frauds
print('ratio :', ratio)

y_train.Fraud *= ratio
y_test.Fraud *= ratio

#Names of all of the features in X_train.
features = X_train.columns.values
print('features : ',features)

#Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
#this helps with training the neural network.
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    # print('feature :',feature , 'mean : ', mean , 'std :', std)
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std

'''
Train the Neural Net
'''

# In [28]
# Split the testing data into validation and testing sets
split = int(len(y_test)/2)
print('split : ', split)

train_x = X_train.as_matrix()
train_y = y_train.as_matrix()
valid_x = X_test.as_matrix()[:split]
valid_y = y_test.as_matrix()[:split]
test_x = X_test.as_matrix()[split:]
test_y = y_test.as_matrix()[split:]

print('type(inputX)  : ', type(train_x))
print('type(X_train) : ', type(X_train))

print('y_train.Normal.value_counts() :',y_train.Normal.value_counts())
print('y_train.Fraud.value_counts() :',y_train.Fraud.value_counts())

print('y_test.Normal.value_counts() :',y_test.Normal.value_counts())
print('y_test.Fraud.value_counts() :',y_test.Fraud.value_counts())

print('C valid_y', np.where(valid_y[:, 0] > 0, 1, 0).sum())
print('C test_y ', np.where(test_y[:, 0] > 0, 1, 0).sum())

print('inputX :',train_x.shape)
print('inputY :',train_y.shape)
print('valid_x :',valid_x.shape)
print('valid_y :',valid_y.shape)
print('test_x :',test_x.shape)
print('test_y :',test_y.shape)

# Confusion Matrix
def get_conf_rate(model, testX, testY):
    # 모델 예측
    predicted_y = model.predict(testX)

    # compare predicted_y & testY
    conf_cnt = collections.defaultdict(int)
    for pr_y, real_y in zip(predicted_y, testY):
        conf = np.argmax(pr_y), np.argmax(real_y)
        conf_cnt[conf] += 1

    # 0번방 == Fraud, 1번방 == Normal
    TP = conf_cnt[(0 ,0)]   # True Positives
    FN = conf_cnt[(1, 0)]   # False Negatives
    TN = conf_cnt[(1, 1)]   # True Negatives
    FP = conf_cnt[(0, 1)]   # False Positives

    # 정분류율
    Acc = (TP + TN) / (TP + TN + FP + FN)
    # 정확도
    try:
        Precision = TP / (TP + FP)
    except:
        Precision = 0
    # 재현율
    try:
        Recall = TP / (TP + FN)
    except:
        Recall = 0
    print('TP :', TP, ' FN :',FN , ' TN :', TN, ' FP :', FP)
    print('Acc(정분류율) :', Acc, 'Precision(정확도) :', Precision, 'Recall(재현율) :', Recall)

    return Acc, Precision, Recall

# Callback Class
class mykerasCB(keras.callbacks.Callback):
    def __init__(self):
        super(mykerasCB, self).__init__()
        self.hist = []

    def on_epoch_end(self, arga, argb):
        print()
        print('arga :', arga)
        print('argb :', argb)
        self.hist.append(get_conf_rate(model, test_x, test_y))

# Number of input nodes.
input_nodes = train_x.shape[1]

# Multiplier maintains a fixed ratio of nodes between each layer.
mulitplier = 1.5

# Number of nodes in each hidden layer
hidden_nodes1 = 18
hidden_nodes2 = round(hidden_nodes1 * mulitplier)
hidden_nodes3 = round(hidden_nodes2 * mulitplier)

# In [31]
# Parameters
training_epochs = 50             # should be 2000, it will timeout when uploading
training_dropout = 0.9          # drop out
display_step = 1                # 10
n_samples = y_train.shape[0]
batch_size = 2048
learning_rate = 0.005           # 하이퍼파라미터

print('======================================================')
print('======================  Keras   ======================')
print('======================================================')

# 모델 구성하기
model = Sequential()
model.add(Dense(hidden_nodes1, input_dim=input_nodes, activation='sigmoid'))
model.add(Dense(hidden_nodes2, activation='tanh'))
model.add(Dense(hidden_nodes3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# Callback Class 선언
cb = mykerasCB()

# 모델 학습과정 설정하기
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

# 모델 학습시키기
hist = model.fit(train_x, train_y, epochs=training_epochs, batch_size=batch_size, callbacks = [cb])

# 모델 평가하기
loss_and_metrics = model.evaluate(valid_x, valid_y, batch_size=batch_size)
print('loss_and_metrics : ' + str(loss_and_metrics))

fig, ax =  plt.subplots()

acc = [h[0] for h in cb.hist]
pre = [h[1] for h in cb.hist]
recall = [h[2] for h in cb.hist]

ax.plot(acc, label='acc')
ax.plot(pre, label='pre')
ax.plot(recall, label='recall')

ax.set_xlabel('epoch')
ax.legend()

plt.show()

print('======================================================')
print('======================  Keras   ======================')
print('======================================================')
