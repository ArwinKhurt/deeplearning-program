import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape(-1, 28*28) / 255.0
test_x = test_x.reshape(-1, 28*28) / 255.0

# Build model manually using TensorFlow Core
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

print("\nTraining TensorFlow model...")
model.fit(train_x, train_y, epochs=5)

print("\nEvaluating...")
loss, acc = model.evaluate(test_x, test_y)
print("TensorFlow Accuracy:", acc)
