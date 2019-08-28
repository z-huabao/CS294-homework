import os
import argparse
import time

from half_cheetah_env import HalfCheetahEnv
from logger import logger
from model_based_rl import ModelBasedRL

parser = argparse.ArgumentParser()
parser.add_argument('question', type=str, choices=('q1, q2, q3'))
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--env', type=str, default='HalfCheetah', choices=('HalfCheetah',))
parser.add_argument('--render', action='store_true')
parser.add_argument('--mpc_horizon', type=int, default=15)
parser.add_argument('--num_random_action_selection', type=int, default=4096)
parser.add_argument('--nn_layers', type=int, default=1)
parser.add_argument('--cem', type=float, default=0., help='Cross Entropy Method')
args = parser.parse_args()

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
exp_name = '{0}_{1}_{2}'.format(
    args.env,
    args.question,
    args.exp_name if args.exp_name else time.strftime("%Y-%m-%d_%H-%M-%S")
)

exp_dir = os.path.join(data_dir, exp_name)
if os.path.exists(exp_dir):
    print('Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(exp_dir))
    key = input("\nTo cover exists! Press 'y' to continue\n")
    if 'y' == key:
        os.system('rm %s/*' % exp_dir)
    else:
        quit(0)

os.makedirs(exp_dir, exist_ok=True)
logger.setup(exp_name, os.path.join(exp_dir, 'log.txt'), 'debug')

env = {
    'HalfCheetah': HalfCheetahEnv()
}[args.env]


mbrl = ModelBasedRL(
    env=env,
    render=args.render,
    mpc_horizon=args.mpc_horizon,
    num_random_action_selection=args.num_random_action_selection,
    nn_layers=args.nn_layers,
    cem=args.cem,
)

run_func = {
    'q1': mbrl.run_q1,
    'q2': mbrl.run_q2,
    'q3': mbrl.run_q3
}[args.question]
run_func()
