
"""
    Experiment script intended to test Behavior Cloning
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gym
import numpy as np
from dart.experiments.tools.expert import load_policy
from dart.experiments.tools import statistics, utils
import argparse
import scipy.stats
import time as timer
import dart.experiments.framework as framework
import pybullet_envs


def main():
    title = 'test_bc'
    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', default="Ant-v2")                         # OpenAI gym environment
    ap.add_argument('--t', default=500, type=int)                     # time horizon
    ap.add_argument('--iters', default=[5], type=int, nargs='+')      # iterations to evaluate the learner on

    args = vars(ap.parse_args())
    args['arch'] = [64, 64]
    args['lr'] = .01
    args['epochs'] = 50

    TRIALS = framework.TRIALS

    test = Test(args)
    start_time = timer.time()
    test.run_trials(title, TRIALS)
    end_time = timer.time()

    print("\n\n\nTotal time: " + str(end_time - start_time) + '\n\n')



class Test(framework.Test):


    def run_iters(self):
        T = self.params['t']

        results = {
            'rewards': [],
            'sup_rewards': [],
            'surr_losses': [],
            'sup_losses': [],
            'sim_errs': []
        }

        snapshots = []
        for i in range(self.params['iters'][-1]):
            print ("\tIteration: " + str(i))

            states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
            states, i_actions = utils.filter_data(self.params, states, i_actions)
            self.lnr.add_data(states, i_actions)

            if ((i + 1) in self.params['iters']):
                snapshots.append((self.lnr.X[:], self.lnr.y[:]))

        for j in range(len(snapshots)):
            X, y = snapshots[j]
            self.lnr.X, self.lnr.y = X, y
            self.lnr.train(verbose=True)
            print ("\nData from snapshot: " + str(self.params['iters'][j]))
            it_results = self.iteration_evaluation()
            
            results['sup_rewards'].append(it_results['sup_reward_mean'])
            results['rewards'].append(it_results['reward_mean'])
            results['surr_losses'].append(it_results['surr_loss_mean'])
            results['sup_losses'].append(it_results['sup_loss_mean'])
            results['sim_errs'].append(it_results['sim_err_mean'])


        for key in results.keys():
            results[key] = np.array(results[key])
        return results




if __name__ == '__main__':
    main()

