## `solve_net.py`的修改

目的：
	方便使用`matplotlib`绘制loss和accuracy的图像

内容：
	 在`train_net`和`test_net`中加入列表记录loss和accuracy，作为返回值返回给`run_mlp.py`

## `run_mlp.py`的修改

目的：
	方便使用`matplotlib`绘制loss和accuracy的图像，使用`softmax`和`FocalLoss`，调整超参数，设置MLP

内容：
	1. 导入`matplotlib`库
	2. 定义四个列表分别记录training和test的loss和accuracy，利用函数返回值进行更新
	3. 使用`matplotlib`绘制图像
	4. 导入`softmax`和`FocalLoss`
	5. 更改`config`中的超参数
	6. 添加隐藏层、激活函数和损失函数，调整隐藏维数

## `loss.py`的修改

调整了`HingeLoss`的$\Delta$

## `layers.py`的修改

增加了一个`softmax`层，用于`HingeLoss`