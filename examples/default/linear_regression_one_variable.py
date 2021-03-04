import time

import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.INFO)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

start_time = time.time()
history = model.fit(celsius_q, fahrenheit_a, epochs=10000, verbose=False)
print(f"Finished training the model. Took {time.time() - start_time} secs")

print(model.predict([100.0]))
