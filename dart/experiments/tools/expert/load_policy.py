"""
Author: John Schulman
"""
from .tf_util import lrelu, function
import pickle
import tensorflow as tf
import numpy as np
from os.path import join as pjoin
import os
import yaml
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from dart.experiments.serialization import load

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "her": HER,
    "sac": SAC,
    "td3": TD3
}

def load_policy(expert_dir, env_id, env, filename):
    with open(pjoin(expert_dir, env_id, "args.yml")) as f:
        algo = yaml.load(f, Loader=yaml.UnsafeLoader)["algo"]
    expert = load(ALGOS[algo], filename, env=env, device="cpu")
    expert_predict = lambda obs, state: expert.predict(obs, state=state, deterministic=True)
    return expert_predict

def load_policy_berkeley(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    # print nonlin_type
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    def build_policy(obs_bo):
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        def apply_nonlin(x):
            if nonlin_type == 'lrelu':
                return lrelu(x, leak=.01) # openai/imitation nn.py:233
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        # print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation

        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            # print "\nW: " + str(W.shape)
            # print "b: " + str(b.shape)
            curr_activations_bd = apply_nonlin(tf.matmul(curr_activations_bd, W) + b)

        # Output layer
        W, b = read_layer(policy_params['out'])
        # print "\nOutput:"
        # print "W: " + str(W.shape)
        # print "b: " + str(b.shape)
        output_bo = tf.matmul(curr_activations_bd, W) + b
        return output_bo

    obs_bo = tf.placeholder(tf.float32, [None, None])
    a_ba = build_policy(obs_bo)
    policy_fn = function([obs_bo], a_ba)
    return policy_fn
