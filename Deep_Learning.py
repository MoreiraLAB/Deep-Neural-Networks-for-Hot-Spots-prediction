import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def expose(input_col):

    return pd.DataFrame(input_col.tolist(),).transpose()

def encode_binary(input_col):

    HS_col, NS_col = [], []
    for class_value in input_col.values:
        if class_value == 1: HS_col.append(1), NS_col.append(0)
        elif class_value == 0: NS_col.append(1), HS_col.append(0)
    output_df = pd.DataFrame()
    output_df[NS], output_df[HS] = HS_col, NS_col
    return output_df

def neural_network(features):
    layer_1 = tf.add(tf.matmul(features, weights['hidden_1']), biases['bias_1'])
    layer_2 = tf.add(tf.matmul(tf.nn.relu(layer_1), weights['hidden_2']), biases['bias_2'])
    layer_3 = tf.add(tf.matmul(tf.nn.relu(layer_2), weights['hidden_3']), biases['bias_3'])
    layer_4 = tf.add(tf.matmul(tf.nn.relu(layer_3), weights['hidden_4']), biases['bias_4'])
    layer_5 = tf.add(tf.matmul(tf.nn.relu(layer_4), weights['hidden_5']), biases['bias_5'])
    out_layer = tf.matmul(tf.nn.relu(layer_5), weights['output'])
    return out_layer

identifiers = pd.read_csv("spoton_clean.csv", sep = ",").iloc[:,0:4]
features = pd.read_csv("spoton_clean.csv", sep = ",").iloc[:,6:]
classes = pd.read_csv("class_clean.csv", sep = ",")
NS, HS = "NS","HS"
encoded_classes = encode_binary(classes)
X_train, X_test, y_train, y_test = train_test_split(features, encoded_classes, test_size=0.3, random_state=42)
num_hidden_1 = 100
num_hidden_2 = 100
num_hidden_3 = 100
num_hidden_4 = 100
num_hidden_5 = 100
num_input = 23
num_classes = 2
display_step = 100
num_steps = 150000
learning_rate = 0.001

weights = {
    'hidden_1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'hidden_2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'hidden_3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
    'hidden_4': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_4])),
    'hidden_5': tf.Variable(tf.random_normal([num_hidden_4, num_hidden_5])),
    'output': tf.Variable(tf.random_normal([num_hidden_5, num_classes])),
    }

biases = {
    'bias_1': tf.Variable(tf.random_normal([num_hidden_1])),
    'bias_2': tf.Variable(tf.random_normal([num_hidden_2])),
    'bias_3': tf.Variable(tf.random_normal([num_hidden_3])),
    'bias_4': tf.Variable(tf.random_normal([num_hidden_4])),
    'bias_5': tf.Variable(tf.random_normal([num_hidden_5])),
    'output': tf.Variable(tf.random_normal([num_classes])),
}



X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("int32", [None, num_classes])
logits = neural_network(X)
loss_calc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_calc)

output_pred = tf.argmax(logits, 1)
true_values = tf.argmax(Y, 1)
correct_pred = tf.equal(output_pred, true_values)
auroc = tf.metrics.auc(labels = true_values, predictions = output_pred),
false_positives = tf.metrics.false_positives(labels = true_values, predictions = output_pred), 
false_negatives = tf.metrics.false_negatives(labels = true_values, predictions = output_pred), 
precision = tf.metrics.precision(labels = true_values, predictions = output_pred), 
recall = tf.metrics.recall(labels = true_values, predictions = output_pred)
accuracy = tf.metrics.accuracy(labels = true_values, predictions = output_pred)

init = tf.global_variables_initializer()    
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    sess.run(tf.initialize_local_variables())
    for step in range(1, num_steps + 1):
        sess.run(train_op, feed_dict={X: X_train, Y: y_train})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_calc, accuracy], feed_dict={X: X_train,
                                                                 Y: y_train})
            print("Step " + str(step) + ", Loss= " + \
                  str(float(loss)) + ", Training Accuracy= " + \
                  str(float(acc[1])))

    print("Optimization Finished!")
    print("Training Accuracy:", \
        sess.run(accuracy, feed_dict={X: X_train,
                                      Y: y_train}))
    print("Training AUROC:", \
        sess.run(auroc, feed_dict={X: X_train,
                                      Y: y_train}))
    print("Training Precision:", \
        sess.run(precision, feed_dict={X: X_train,
                                      Y: y_train}))
    print("Training Recall:", \
        sess.run(recall, feed_dict={X: X_train,
                                      Y: y_train}))

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: X_test,
                                      Y: y_test}))
    print("Testing AUROC:", \
        sess.run(auroc, feed_dict={X: X_test,
                                      Y: y_test}))
    print("Testing Precision:", \
        sess.run(precision, feed_dict={X: X_test,
                                      Y: y_test}))
    print("Testing Recall:", \
        sess.run(recall, feed_dict={X: X_test,
                                      Y: y_test}))

