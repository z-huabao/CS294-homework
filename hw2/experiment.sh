#/usr/bin/env bash

## exp1
#python3 train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
#python3 train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
#python3 train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
#python3 train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
#python3 train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
#python3 train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na #--render

#python3 plot.py ./data/sb* -s ./data/CartPole_small_batch.png
#python3 plot.py ./data/lb* -s ./data/CartPole_large_batch.png


## exp2
#b=8000
#lr=1e-1
#python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 \
    #-l 2 -s 64 -b $b -lr $lr -rtg --exp_name ip_b${b}_r${lr} \
    #--loop --render
#python3 plot.py ./data/ip_b*_r* -s ./data/InvertedPendulum.png

## exp3
#ll_dir=/usr/local/lib/python3.5/dist-packages/gym/envs/box2d/
#cp ${ll_dir}lunar_lander.py ${ll_dir}lunar_lander-old.py && cp ./lunar_lander.py ${ll_dir}lunar_lander.py
#pip3 install box2d box2d-kengz
#python3 train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n \
    #100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005 \
    #--render
#python3 plot.py ./data/ll_b*_r* -s ./data/LunarLanderContinuous.png

## exp4-1
#for b in 10000 30000 50000; do
#for lr in 0.005 0.01 0.02; do
    #python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 \
        #-s 32 -b $b -lr $lr -rtg --nn_baseline --exp_name hc_search/hc_b${b}_r${lr} \
        #--loop --render
#done;
#done;
#python3 plot.py ./data/hc_search/hc_b*_r* -s ./data/hc_search/hc_search.png

## exp4-2
#b=30000
#lr=0.02
#python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
    #-l 2 -s 32 -b ${b} -lr ${lr} --exp_name hc_b${b}_r${lr}

#python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
    #-l 2 -s 32 -b ${b} -lr ${lr} -rtg --exp_name hc_rtg_b${b}_r${lr}

#python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
    #-l 2 -s 32 -b ${b} -lr ${lr} --nn_baseline --exp_name hc_bl_b${b}_r${lr}

#python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
    #-l 2 -s 32 -b ${b} -lr ${lr} -rtg --nn_baseline --exp_name hc_rtg_bl_b${b}_r${lr}

#python3 plot.py ./data/hc_*b*_r* -s ./data/HalfCheetah.png

## exp4-3
b=10000
lr=0.02
python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
    -l 2 -s 8 -b ${b} -lr ${lr} -rtg --nn_baseline --exp_name hc_s8_rtg_bl_b${b}_r${lr}
python3 plot.py ./data/hc_*b*_r* -s ./data/HalfCheetah.png
