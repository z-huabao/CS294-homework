## actor-critic

这个方法是前面策略梯度的改进，前面提到用reward-to-go来统计reward作为q值，这里希望用一个critic网络来学习q。

###critic

`critic`输入observation，输出对单个q值，这里q值的含义好像不一样了，我理解是代表对今后奖励的期望（不再像之前一样是具体路径的真实奖励，而是被今后各种可能路径的奖励给平均了，从下面策略来看是依概率加权平均的）；

######如果游戏结束了，那critic的预测是无效的，应当看做0。

对当前的observation有好几种q，下面从1到3，路径越来越明确，q越来越具体：

    1，critic直接预测当前observation；
    2，进行一个action，得到一个reward，reward+下一个observation的critic预测；
    3，reward-to-go；

上述2的q比1更具体，所以1的q为predict，用2的q作为ground-truth拟合（感觉有点问题）。
`critic`的loss用l2，或smooth-l1。

###actor

上述2的q减1的q叫做advantage：`adv = q' - q`，作为`actor`动作好坏的评判标准。

advantage和reward近似却不一样：

    reward是短视的，只关注当前得分；advantage是和整体得分关联的。

`actor`和策略梯度几乎是一样的，loss和之前也类似：`loss = -(prob * adv)`

训练过程：

    收集path；
    训练critic；
    用critic处理path得到adv_n；
    训练actor;
    循环。。。

### Q2.1 Sanity check with Cartpole

<div align=center> <img src="./data/CartPole-v0(ntu-ngsptu).png" height="300px"> </div>

### Q2.2 Run actor-critic with more difficult tasks

<div align=center>
<img src="./data/HalfCheetah-v2(ntu-ngsptu).png.png" height="300px">
<img src="./data/InvertedPendulum-v2(ntu-ngsptu).png" height="300px">
</div>

## DQN

###Q-learning

Q-learning假设有一个超大的表格，表格行是observation，列是action，数值是observation和action对应的q值。随着对环境的探索越来越多，表格也会越来越大，无穷大。然后测试时假如碰到了同样的状态后就选择q值比较大的action就是最优的路径了。

在探索（explore）中一开始完全使用随机action，后来表格丰富后则大概率选择q值大的action，小概率用随机action。

###Deep Q Network

DQN就是用深度神经网络来代替表格，网络输入observation，输出每个action可能的q值。

    表格可能无穷大，神经网络的大小总是有限的；
    对于从未探索过的observation来说，表格无法提供策略，但是神经网络始终有q值输出；
    DQN和actor-critic的critic很像，不过critic输出一个q，DQN输出n_actions个q。

数据：

    训练神经网络需要收集数据，这些数据其实也是上述表格，但大小有限；
    要大批量训练数据才有好的效果，所以要先随机收集一波数据，然后训练；
    然后又收集一小波新的（训练后可以探索新的环境）和原来的混合一起训练。。。一直反复；
    这样就需要一个数据池，每次有放回的取数据，还要时常添加更新；

训练：

    这里的q也不用reward-to-go，而是和上面actor-critic的critic一样，loss也一样。
    因为输出q的数量有n_actions个，但实际只执行了一个action，所以也只计算那个action对应q的loss。

------------

# CS294-112 HW 3: Q-Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.
