一个常用的神经网络模板


* checkpoints/ 
用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练。

* data/
数据相关操作，包括数据预处理、dataset实现等

    * \_\_init\_\_.py
    from .dataset import Dataset

    * train
    训练集数据

    * test:
    测试集数据

    * dataset.py:
    class Dataset(torch.utils.data.Dataset)

* models/
模型定义，可以有多个模型，一个模型对应一个文件

    * \_\_init\_\_.py
    from .BasicModule import BasicModule

    * Net.py
    class Net(BasicModule)

    * BasicModule.py
    class BasicModule(torch.nn.Module)

* utils/
可能用到的工具函数，本次实验中主要封装了可视化工具
    * \_\_init\_\_.py
    from .visualizer import Visualizer
    * visualizer.py
    class Visualizer()
* config.py
配置文件，所有可配置的变量都集中于此，并提供默认值
* main.py
主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数。
* requirements.txt
程序依赖的第三方库
* README.md
