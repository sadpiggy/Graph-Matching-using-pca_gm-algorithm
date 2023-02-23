# Graph-Matching-using-pca_gm-algorithm
+ 我使用的算法为pca_gm算法[[“Wang et al. Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach. TPAMI 2020.”](https://ieeexplore.ieee.org/abstract/document/9128045/)]
+ 第一步，通过一个卷积网络提取特征

+ 第二步，特征输入到embedding layers。其中,embedding layers由几个intra-graph embedding layer和一个cross graph embedding layer组成

+ 第三步，第二步中的Embedding layers输出一个关系矩阵，通过一个sinkhorn matching layer得到匹配矩阵
