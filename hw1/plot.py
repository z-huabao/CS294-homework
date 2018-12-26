#/usr/bin/env python3
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from random import randint


def load_base_value():
    data = pd.read_csv('./expert_data/meanReturns.txt', header=None)
    data = dict(zip(data[0], data[1]))
    return data

def plot_data(data, value="", base_value=None):
    sns.set(style="darkgrid", font_scale=1.1)
    for k, d in data.items():
        avgR = d['AverageReturn']
        stdR = d['StdReturn']
        stdMin = avgR - stdR
        stdMax = avgR + stdR
        color = "00000" + hex(randint(0, 256*256*256))[2:]
        color = "#" + color[-6:]
        sns.tsplot([stdMin, stdMax], condition=k, color=color)

    if base_value:
        n = len(avgR)
        plt.plot(range(n), base_value * np.ones(n), "k:", linewidth=1)
        plt.text(n-1, base_value, "base_line", fontsize=10)

def get_datasets(fpath, condition=None, env=''):
    data = {}
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files and env in root:
            exp_name = condition or root.split('/')[-1].split('_2018')[0]
            log_path = os.path.join(root,'log.txt')
            experiment_data = pd.read_table(log_path)
            data[exp_name] = experiment_data
    return data


def main():
    import argparse
    # usage: python3 plot.py ./log --env=Ant-v2
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    parser.add_argument('--env', default='Hopper-v2')
    parser.add_argument('--save', default=True)
    args = parser.parse_args()

    base_value = load_base_value().get(args.env)

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data = get_datasets(logdir, legend_title, env=args.env)
    else:
        for logdir in args.logdir:
            data = get_datasets(logdir, env=args.env)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value, base_value)

    plt.xlabel('epochs')
    plt.ylabel('rewards')
    if args.save:
        plt.savefig('./log/%s.png' % args.env)
    plt.show()

if __name__ == "__main__":
    main()
