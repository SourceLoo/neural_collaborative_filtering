## Dataset.py

###稀疏矩阵的表示方式：

- Dictionary of keys (DOK)；
字典，key: (r,c); value: val
dok matrix 一般用于快速构建sparse matrix

- Coordinate list (COO)
list, 每个item是(r,c,v)

- Compressed sparse row (CSR)
三个array：A;AI;AJ；

其中A存储所有非零元素值（按行为主），AJ存储每个值对应的所在列号，这两者长度相同；
AI存储前i行的非零值的个数，长度为rows；即可确定每个元素的所在行列。



xrange is faster than range





## EvaluateTopKofRange.py
存储 parameter的model file在哪


## GMFlogistic.py
keras是高层的api，可以选择backend为 tf、theano、cntk

1 epoch，指所有instance，bp训练一遍;
batch size，指mini-batch中，有多少instance参与此次bp，
iteration，一次bp，为一次迭代；
注意：一次迭代中，所有instance的参数都会改变，但是代价和是batch size的代价和

故一次epoch可以有多次iteration；

特别的：
GD的batch size为全部instance；
SGD的batch size为1；
mini-batch SG的batch size为(1,n)

batch size 大，一次iteration越耗内存，越慢，但是gradient越准确；


https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network



item的编码打乱了