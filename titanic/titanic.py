# url -> https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic/notebook

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Feature Engineering
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from collections import namedtuple
from sklearn.preprocessing import Binarizer

# load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

def nan_padding(data, columns):
    for column in columns:
        # 누락된 값(nan) 초기화 하기
        imputer = Imputer()
        data[column] = imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

nan_columns = ["Age", "SibSp", "Parch"]

train_data = nan_padding(train_data, nan_columns)
test_data  = nan_padding(test_data,  nan_columns)

#save PassengerId for evaluation
test_passenger_id = test_data["PassengerId"]

# 필요 없는 컬럼 drop
def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]
train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)

# print(train_data.head())

# 범주 형 변수를 더미 변수 / 지시 변수로 변환 -> pd.get_dummies
def dummy_data(data, columns):
    for column in columns:
        # print('pd.get_dummies(data[column], prefix=column) :', pd.get_dummies(data[column], prefix=column))
        # data concat
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        # Pclass 컬럼 drop
        data = data.drop(column, axis=1)
    return data

dummy_columns = ["Pclass"]
train_data = dummy_data(train_data, dummy_columns)
test_data  = dummy_data(test_data, dummy_columns)

# 성별(male, female) int형으로 변환 male 1, female 0
def sex_to_int(data):
    le = LabelEncoder()
    le.fit(['male','female'])
    data["Sex"] = le.transform(data["Sex"])
    return data

train_data = sex_to_int(train_data)
test_data  = sex_to_int(test_data)

# print(train_data.head())

# Age 컬럼 정규화
def normalize_age(data):
    scaler = MinMaxScaler()
    # print('scaler.fit_transform(data["Age"].values.reshape(-1, 1) :', scaler.fit_transform(data["Age"].values.reshape(-1, 1)))
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1, 1))
    return data

train_data = normalize_age(train_data)
test_data  = normalize_age(test_data)

# print(train_data.head())

# data split
def split_valid_test_data(data, fraction=(1 - 0.8)):
    # Survived 컬럼만으로 구성된 data_y 구성
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    # data_x의 Survived 컬럼 drop
    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x, train_y, valid_x, valid_y

# data split 적용
train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)

print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))

print("valid_x:{}".format(valid_x.shape))
print("valid_y:{}".format(valid_y.shape))

print('train_x.columns :', train_x.columns.values)

# Build Neural Network
def build_neural_network(hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.Variable(True, dtype=tf.bool)

    initializer = tf.contrib.layers.xavier_initializer()
    # layer 구성
    fc = tf.layers.dense(inputs, hidden_units, activation=None, kernel_initializer=initializer)
    fc = tf.layers.batch_normalization(fc, training=is_training)
    # relu 활성화 함수 적용
    fc = tf.nn.relu(fc)

    # 손실 비용 계산
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)

    # optimizer 설정
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 정확도 계산
    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes
    export_nodes = ['inputs', 'labels', 'learning_rate', 'is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

model = build_neural_network()

# batch 적용하기
def get_batch(data_x, data_y, batch_size=32):
    batch_n = len(data_x)//batch_size
    for i in range(batch_n):
        batch_x = data_x[i * batch_size : (i+1) * batch_size]
        batch_y = data_y[i * batch_size : (i+1) * batch_size]

        yield batch_x, batch_y

# epoch, 하이파라미터, train, validation 초기화
epochs = 200
train_collect = 50
train_print = train_collect * 2

learning_rate_value = 0.001
batch_size = 16

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 0
    for e in range(epochs):
        for batch_x, batch_y in get_batch(train_x, train_y, batch_size):
            # 학습하기
            iteration += 1
            feed = {model.inputs: train_x,
                    model.labels: train_y,
                    model.learning_rate: learning_rate_value,
                    model.is_training: True
                    }

            train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)

            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print == 0:
                    print("Epoch: {}/{}".format(e + 1, epochs),
                          "Train Loss: {:.4f}".format(train_loss),
                          "Train Acc: {:.4f}".format(train_acc))

                # validation
                feed = {model.inputs: valid_x,
                        model.labels: valid_y,
                        model.is_training: False
                        }

                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)

                if iteration % train_print == 0:
                    print("Epoch: {}/{}".format(e + 1, epochs),
                          "Validation Loss: {:.4f}".format(val_loss),
                          "Validation Acc: {:.4f}".format(val_acc))

    saver.save(sess, "./titanic.ckpt")

model = build_neural_network()
restorer = tf.train.Saver()

# save 읽어와서 test_data 적용
with tf.Session() as sess:
    restorer.restore(sess, "./titanic.ckpt")
    feed = {
        model.inputs : test_data,
        model.is_training : False
    }
    test_predict = sess.run(model.predicted, feed_dict=feed)

# test 예측값 출력 ( Survived 확률 )
print('test_predict[:10] :', test_predict[:10])

# 0.5 보다 큰 값은 1로 매핑, 작은 값은 0으로 매핑
binarizer = Binarizer(0.5)
test_predict_result = binarizer.fit_transform(test_predict)
# int type으로 변환
test_predict_result = test_predict_result.astype(np.int32)

print('test_predict_result[:10] :', test_predict_result[:10])

# test의 결과를 csv 파일생성
passenger_id = test_passenger_id.copy()
evaluation = passenger_id.to_frame()
print('evaluation 1 :', evaluation[:10])
evaluation["Survived"] = test_predict_result
print('evaluation 2 :', evaluation[:10])
evaluation["test_predict"] = test_predict
print('evaluation 3 :', evaluation[:10])

# csv 파일 저장
evaluation.to_csv("evaluation_submission.csv", index=False)

# plt show
plt.plot(x_collect, train_loss_collect, "r--", label='train_loss_collect')
plt.plot(x_collect, valid_loss_collect, "g^",  label='valid_loss_collect')
plt.xlabel("epoch")
plt.legend()
plt.show()

plt.plot(x_collect, train_acc_collect, "r--", label='train_acc_collect')
plt.plot(x_collect, valid_acc_collect, "g^",  label='valid_acc_collect')
plt.xlabel("epoch")
plt.legend()
plt.show()