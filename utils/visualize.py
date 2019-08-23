import visdom
import time
import numpy as np

class Visualier(object):

    '''
    封装了visdom的基本操作，但仍然可以通过self.vis.function
    或者self.function调用原生的visdom接口
    '''

    def __init__(self, env = 'default', **kwargs):

        self.vis = visdom.Visdom(env = env, **kwargs)

        # 保存('loss', 23)  即loss的第23个节点
        self.index = {}
        self.log_text = ''

    def reinit(self, env = 'default', **kwargs):

        '''
        修改visdom的配置
        '''

        self.vis = visdom.Visdom(env = env, **kwargs)
        return self

    def __getattr__(self, name):

        '''
        self.function等价于self.vis.function
        '''

        return getattr(self.vis,name)

