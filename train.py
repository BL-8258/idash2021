import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow import keras
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D,GRU,SimpleRNN
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation, Flatten, Reshape, Dense,LSTM,Bidirectional
from tensorflow.keras.models import Model, Sequential 

import collections
import tensorflow_federated as tff
import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow.keras.regularizers import l1, l2, l1_l2
import nest_asyncio
nest_asyncio.apply()
import math
import pandas as pd
from checkpoint_manager import FileCheckpointManager

import shutil

tf.compat.v1.flags.DEFINE_float('noise_multiplier', 0.5,
                   'Ratio of the standard deviation to the clipping norm')
tf.compat.v1.flags.DEFINE_integer('batch_size', 30, 'Batch size')
tf.compat.v1.flags.DEFINE_integer('epochs', 20, 'Number of epochs')
tf.compat.v1.flags.DEFINE_integer('rounds', 10, 'Number of rounds')
tf.compat.v1.flags.DEFINE_string('model_dir', '...', 'Model directory')

FLAGS = tf.compat.v1.flags.FLAGS

DELTA=10e-3

def preprocess(dataset,datatype='train'):
    def batch_format_fn(element1,element2):
        return collections.OrderedDict(
            x=tf.cast(tf.reshape(element1, [-1, 1874, 1]),tf.float32),
            y=tf.cast(element2,tf.float32))
    if datatype=='train':
        return dataset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).repeat(FLAGS.epochs).map(batch_format_fn)
    else:
        return dataset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).repeat(FLAGS.epochs).map(batch_format_fn)
		
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1874, 1)),
        tf.keras.layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l1_l2(0.0001)),
        tf.keras.layers.AveragePooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Reshape([100,1]),
        tf.keras.layers.Bidirectional(LSTM(units = 80)),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
	
def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    #print(keras_model.summary())
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=train_dataset[0].element_spec,
        loss=tf.losses.BinaryCrossentropy(name='binary_crossentropy'),
         metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.MeanSquaredError()])
		 
		 
def make_optimizer_class(cls):
  """Constructs a DP optimizer class from an existing one."""
  parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
  child_code = cls.compute_gradients.__code__
  GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip', 'stddev'])
    
    def __init__(
        self,
        dp_sum_query,
        num_microbatches=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        dp_sum_query: DPQuery object, specifying differential privacy
          mechanism to use.
        num_microbatches: How many microbatches into which the minibatch is
          split. If None, will default to the size of the minibatch, and
          per-example gradients will be computed.
        unroll_microbatches: If true, processes microbatches within a Python
          loop instead of a tf.while_loop. Can be used if using a tf.while_loop
          raises an exception.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._dp_sum_query = dp_sum_query
      self._num_microbatches = num_microbatches
      self._global_state = self._dp_sum_query.initial_global_state()
      # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
      # Beware: When num_microbatches is large (>100), enabling this parameter
      # may cause an OOM error.
      self._unroll_microbatches = unroll_microbatches

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None,
                          curr_noise_mult=0,
                          curr_norm_clip=1):

      self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip, 
                                                           curr_norm_clip*curr_noise_mult)
      self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip, 
                                                                curr_norm_clip*curr_noise_mult)
      

      # TF is running in Eager mode, check we received a vanilla tape.
      if not gradient_tape:
        raise ValueError('When in Eager mode, a tape needs to be passed.')

      vector_loss = loss()
      if self._num_microbatches is None:
        self._num_microbatches = tf.shape(input=vector_loss)[0]
      sample_state = self._dp_sum_query.initial_sample_state(var_list)
      microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
      sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

      def process_microbatch(i, sample_state):
        """Process one microbatch (record) with privacy helper."""
        microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
        grads = gradient_tape.gradient(microbatch_loss, var_list)
        sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
        return sample_state
    
      for idx in range(self._num_microbatches):
        sample_state = process_microbatch(idx, sample_state)

      if curr_noise_mult > 0:
        grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
      else:
        grad_sums = sample_state

      def normalize(v):
        return v / tf.cast(self._num_microbatches, tf.float32)

      final_grads = tf.nest.map_structure(normalize, grad_sums)
      grads_and_vars = final_grads#list(zip(final_grads, var_list))
    
      return grads_and_vars

  return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
  """Constructs a DP optimizer with Gaussian averaging of updates."""

  class DPGaussianOptimizerClass(make_optimizer_class(cls)):
    """DP subclass of given class cls using Gaussian averaging."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        ledger=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)

      if ledger:
        dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
                                                      ledger=ledger)

      super(DPGaussianOptimizerClass, self).__init__(
          dp_sum_query,
          num_microbatches,
          unroll_microbatches,
          *args,
          **kwargs)

    @property
    def ledger(self):
      return self._dp_sum_query.ledger

  return DPGaussianOptimizerClass

# Read data
file_1 = 'Dataset/party_1.csv'
file_2 = 'Dataset/party_2.csv'
party1 = pd.read_csv(file_1, sep=',', index_col=None, header=None, low_memory=False)
party2 = pd.read_csv(file_2, sep=',', index_col=None, header=None, low_memory=False)

data_x1 = party1.iloc[1:,2:-1].values.astype('float')
data_x2 = party2.iloc[1:,2:-1].values.astype('float')


data_y1 = party1.iloc[1:,-1].values.astype('float')
data_y2 = party2.iloc[1:,-1].values.astype('float')


sample_size = data_x1.shape[0]+data_x2.shape[0]

print('Party 1 has %d samples. Party2 has %d samples' %(data_x1.shape[0],data_x2.shape[0]))

train_dataset=[]
train_dataset.append(preprocess(tf.data.Dataset.from_tensor_slices((data_x1,data_y1)),'train'))
train_dataset.append(preprocess(tf.data.Dataset.from_tensor_slices((data_x2,data_y2)),'train'))


AdamOptimizer = tf.compat.v1.train.AdamOptimizer
DPAdamOptimizer = make_gaussian_optimizer_class(AdamOptimizer)

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
#    client_optimizer_fn= lambda: DPKerasAdamOptimizer(l2_norm_clip=100,noise_multiplier=0),
#     client_optimizer_fn= lambda: tf.keras.optimizers.Adam(),
    client_optimizer_fn=lambda: DPAdamOptimizer(l2_norm_clip = 100, noise_multiplier = FLAGS.noise_multiplier),
    server_optimizer_fn= lambda: tf.keras.optimizers.Adam(),
    use_experimental_simulation_loop=True)

	
state = iterative_process.initialize()

acc_list = []
loss_list = []
for round_num in range(FLAGS.rounds):
  state, metrics = iterative_process.next(state, train_dataset)
  print('round {:2d}, metrics={}'.format(round_num, metrics['train']))
  
eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            sample_size, FLAGS.batch_size, FLAGS.noise_multiplier, FLAGS.epochs, DELTA)

if os.path.exists(FLAGS.model_dir):
	shutil.rmtree(FLAGS.model_dir, ignore_errors=True)		
ckpt_manager = FileCheckpointManager(FLAGS.model_dir)
# ckpt_manager.save_checkpoint(ServerState.from_anon_tuple(state), round_num=10)
ckpt_manager.save_checkpoint(state, round_num=FLAGS.rounds)

