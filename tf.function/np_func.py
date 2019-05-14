import numpy as np
import tensorflow as tf

@tf.function
def tf_func(x):
    # add to auto graph
    x = x + 1
    return x

def np_func(x):
    x = x > 0
    print(type(x))
    x = np.array(x, dtype=np.int32)
    x = x * 5
    return x


a = tf.Variable([[2, 2],[-3,-2]], dtype=tf.int32)
output = tf_func(a)
print(output)
np_a = a.numpy()
print(np_a)
# print(tf.autograph.to_code(tf_func.python_function))

# tf.py_func is deprecated in 2.0
# not differentiable
y = tf.numpy_function(np_func, [a], tf.int32)
print('----------------------------')
# differentiable
y2= tf.py_function(np_func, [a], tf.int32)
print(y)
print(y2)
