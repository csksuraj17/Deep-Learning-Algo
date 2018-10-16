import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train/255.
x_test = x_test/ 255.

model = tf.keras.models.Sequential([Flatten(),
  Dense(1024, activation=tf.nn.relu),
  Dropout(0.2),
  Dense(512, activation=tf.nn.relu),
  Dropout(0.2),
  Dense(256, activation=tf.nn.relu),
  Dropout(0.2),
  Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
