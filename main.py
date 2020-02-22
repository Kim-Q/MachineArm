# -*- coding: utf-8 -*-


from env import ArmEnv
from rl import DDPG

MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim  # neural network的输入信息
a_dim = env.action_dim  # 机械手臂的动作数量
a_bound = env.action_bound  # 转动角度的范围

# set RL method
rl = DDPG(a_dim, s_dim, a_bound)


def train():
    """
    训练
    :return:
    """
    for i in range(MAX_EPISODES):
        s = env.reset()  # 重置
        for j in range(MAX_EP_STEPS):
            env.render()  # 可视化

            a = rl.choose_action(s)  # 输入神经网络s，输出a

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)  # 放到记忆库中进行离线学习

            # 如果记忆库装满了，rl开始学习
            if rl.memory_full:
                rl.learn()

            s = s_  # 进行下一个state学习

    rl.save()


def eval():
    """
    测试
    :return:
    """
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)  # 适应屏幕的刷新频率，方便观察
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()
