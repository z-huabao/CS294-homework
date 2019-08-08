#!/usr/bin/env bash

#python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 \
    #--exp_name 1_1 -ntu 1 -ngsptu 1

#python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 \
    #--exp_name 100_1 -ntu 100 -ngsptu 1

#python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 \
    #--exp_name 1_100 -ntu 1 -ngsptu 100

#python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 \
    #--exp_name 100_100 -ntu 100 -ngsptu 100

#python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 \
    #--exp_name 20_5 -ntu 20 -ngsptu 5

#python3 plot.py \
    #data/ac_1_1_CartPole-v0_23-07-2019_15-46-07 \
    #data/ac_100_1_CartPole-v0_23-07-2019_15-47-28 \
    #data/ac_1_100_CartPole-v0_23-07-2019_15-48-33 \
    #data/ac_10_10_CartPole-v0_23-07-2019_15-50-21 \
    #data/ac_20_5_CartPole-v0_23-07-2019_19-37-47 \
    #--legend ntu1_ngsptu1 ntu100_ngsptu1 ntu1_ngsptu100 ntu10_ngsptu10 ntu20_ngsptu5 \
    #-s 'data/CartPole-v0(ntu-ngsptu).png'


#------------------------------------------

#python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 \
    #--discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 \
    #--exp_name l2s64 -ntu 10 -ngsptu 10

#python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 \
    #--discount 0.95 -n 100 -e 3 -l 3 -s 64 -b 5000 -lr 0.01 \
    #--exp_name l3s64 -ntu 10 -ngsptu 10

#python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 \
    #--discount 0.95 -n 100 -e 3 -l 3 -s 96 -b 5000 -lr 0.01 \
    #--exp_name l3s96 -ntu 10 -ngsptu 10

#python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 \
    #--discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 \
    #--exp_name 50_10 -ntu 50 -ngsptu 10

#python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 \
    #--discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 \
    #--exp_name 10_50 -ntu 10 -ngsptu 50

python3 plot.py \
    data/ac_l2s64_InvertedPendulum-v2_24-07-2019_20-08-22 \
    data/ac_l3s64_InvertedPendulum-v2_24-07-2019_21-20-45 \
    data/ac_l3s96_InvertedPendulum-v2_24-07-2019_21-25-03 \
    data/ac_50_10_InvertedPendulum-v2_25-07-2019_09-29-23 \
    data/ac_10_50_InvertedPendulum-v2_25-07-2019_09-36-56 \
    --legend l2s64_ntu10_ngstpu10 l3s64 l3s96 ntu50_ngstpu10 ntu10_ngstput50 \
    -s 'data/InvertedPendulum-v2(ntu-ngsptu).png'

#------------------------------------------

#python3 train_ac_f18.py HalfCheetah-v2 -ep 150 \
    #--discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 \
    #--exp_name l2s32 -ntu 10 -ngsptu 10

#python3 train_ac_f18.py HalfCheetah-v2 -ep 150 \
    #--discount 0.90 -n 100 -e 3 -l 3 -s 32 -b 30000 -lr 0.02 \
    #--exp_name l3s32 -ntu 10 -ngsptu 10

#python3 train_ac_f18.py HalfCheetah-v2 -ep 150 \
    #--discount 0.90 -n 100 -e 3 -l 3 -s 64 -b 30000 -lr 0.02 \
    #--exp_name l3s64 -ntu 10 -ngsptu 10

#python3 train_ac_f18.py HalfCheetah-v2 -ep 150 \
    #--discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 \
    #--exp_name 50_10 -ntu 50 -ngsptu 10

#python3 train_ac_f18.py HalfCheetah-v2 -ep 150 \
    #--discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 \
    #--exp_name 10_50 -ntu 10 -ngsptu 50


#python3 plot.py \
    #data/ac_l2s32_HalfCheetah-v2_24-07-2019_20-12-07 \
    #data/ac_l3s32_HalfCheetah-v2_24-07-2019_21-30-35 \
    #data/ac_l3s64_HalfCheetah-v2_24-07-2019_21-53-00 \
    #data/ac_50_10_HalfCheetah-v2_25-07-2019_08-12-55 \
    #data/ac_10_50_HalfCheetah-v2_25-07-2019_08-44-49 \
    #--legend l2s32_ntu10_ngstpu10 l3s32 l3s64 ntu50_ngstpu10 ntu10_ngstpu50 \
    #-s 'data/HalfCheetah-v2(ntu-ngsptu).png'


#python3 run_dqn_atari.py
