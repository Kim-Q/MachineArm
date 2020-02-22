# -*- coding: utf-8 -*-
import numpy as np
import pyglet


class ArmEnv(object):
    """

    """
    viewer = None
    dt = 0.1  # refresh rate，单位时间
    action_bound = [-1, 1]  # 转动角每一步可转动的角度范围
    goal = {'x': 100., 'y': 100., 'l': 40}  # 机械臂末端的位置和大小
    state_dim = 2  # 2个可观测的手臂转动角
    action_dim = 2  # 可动关节数

    def __init__(self):
        # 产生一个2X2的表格
        self.arm_info = np.zeros(2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100  # 手臂长度
        self.arm_info['r'] = np.pi / 6  # 转动角的度数

    def step(self, action):
        done = False  # 检查是否结束
        r = 0.  # reward，后面还会修改
        action = np.clip(action, *self.action_bound)  # 把超出的部分截到对应的点，防止代码出现异常，例如传入值超出阈值的情况
        self.arm_info['r'] += action * self.dt  # 相当于随机游走累积的过程
        self.arm_info['r'] %= np.pi * 2  # normalize

        # state
        s = self.arm_info['r']

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)  # 机械臂的末端

        # done and reward 判断回合是否结束，若结束奖励r=1.，若未结束则继续
        if (self.goal['x'] - self.goal['l'] / 2 < finger[0] < self.goal['x'] + self.goal['l'] / 2
        ) and (self.goal['y'] - self.goal['l'] / 2 < finger[1] < self.goal['y'] + self.goal['l'] / 2):
            done = True
            r = 1.

        return s, r, done

    def reset(self):
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)  # 每次更新输入两个随机的初始角度
        return self.arm_info['r']

    def render(self):
        """
        可视化，调用Viewer中的render
        :return:
        """
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2) - 0.5  # 临时的初始值设置，后面不会用


class Viewer(pyglet.window.Window):
    bar_thc = 5  # bar/机械臂的宽度

    def __init__(self, arm_info, goal):
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)  #
        # 窗口的大小，不改变窗口大小，窗口名称，窗口刷新频率不与窗口一致(?)
        pyglet.gl.glClearColor(1, 1, 1, 1)  # 背景颜色
        self.arm_info = arm_info  # 两节机械手臂信息
        self.center_coord = np.array([200, 200])  # 机械臂头部的位置确定

        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        # point是机械手臂去接触的物体
        self.point = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,  # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))  # color
        # arm分别是两节手臂
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # location
                     250, 400,
                     260, 400,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))  # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        """
        可视化，刷新并呈现在屏幕上
        :return:
        """
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        """
        刷新手臂等位置
        :return:
        """
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        """
        更新手臂的位置信息，一堆三角函数计算
        :return:
        """
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))


if __name__ == "__main__":
    env = ArmEnv()
    while True:
        s = env.reset()
        for i in range(40):
            env.render()
            env.step(env.sample_action())
