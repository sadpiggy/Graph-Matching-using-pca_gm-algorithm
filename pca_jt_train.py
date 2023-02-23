# 本代码基于https://pygmtools.readthedocs.io/en/latest/auto_examples/jittor/plot_deep_image_matching.html#matching-images-with-other-neural-networks
# 使用的算法图匹配算法pca_gm出自“Wang et al. Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach. TPAMI 2020.”
# 使用了了pygmtools工具，https://pygmtools.readthedocs.io/en/latest/
# 作者：20级ACM班王崑运
import jittor as jt # jittor backend
from jittor import Var, models, nn
from jittor.dataset import Dataset
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
import numpy as np
from PIL import Image
import pathlib
pygm.BACKEND = 'jittor' # set default backend for pygmtools
jt.flags.use_cuda = jt.has_cuda

# 数据集加载

obj_resize = (256, 256)


def getPaths(rootPath="data/WillowObject/WILLOW-ObjectClass"):
    rootPath = pathlib.Path(rootPath)
    categories = ["Car","Duck","Face","Motorbike","Winebottle"]
    trainImgPaths = []
    testImgPaths = []
    trainKptPaths = []
    testKptPaths = []
    
    for category in categories:
        category_image_paths = list(rootPath.glob(category+'/*.png'))
        category_image_paths = [str(path) for path in category_image_paths ]
        category_mat_paths = list(rootPath.glob(category+'/*.mat'))
        category_mat_paths = [str(path) for path in category_mat_paths ]
        len = category_image_paths.__len__()
        trainLen = (int)(len*0.8)
        
        trainImgPaths.append(category_image_paths[0:24])
        trainKptPaths.append(category_mat_paths[0:24])
        testImgPaths.append(category_image_paths[24:30])
        testKptPaths.append(category_mat_paths[24:30])
    return trainImgPaths,trainKptPaths,testImgPaths,testKptPaths
    

def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = jt.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

class MyDataset(Dataset):
    def __init__(self,imgPaths,kptPaths,batch_size=1,shuffle=False):
        # 需要shuffle吗？
        super(MyDataset,self).__init__()
        super(MyDataset,self).set_attrs(batch_size=batch_size,shuffle=shuffle)
        self.batch_size = batch_size
        self.lens = []
        self.nowLen = 0
        self.total_len = 0
        self.categories = ["Car","Duck","Face","Motorbike","Winebottle"]
        self.nowCategoty = 0
        self.kptPaths = kptPaths
        self.imgPaths = imgPaths
       
        for category_paths in imgPaths:
            self.lens.append(len(category_paths))
            self.total_len += len(category_paths)
        print(self.lens)
        
    
    def setCategory(self,category:int):
        self.nowCategoty = category
        self.nowLen = self.lens[category]
        
    def getImage(self,imgPath,kptPath):#多半是因为我的数据集加载方式有问题
        img = Image.open(imgPath)
        kpt = jt.Var(sio.loadmat(kptPath)['pts_coord'])
        oralImageSize0 = img.size[0]
        oralImageSize1 = img.size[1]
        kpt[0] = kpt[0] * obj_resize[0] / oralImageSize0
        kpt[1] = kpt[1] * obj_resize[1] / oralImageSize1
        A = (delaunay_triangulation(kpt))
        img = img.resize(obj_resize, resample=Image.BILINEAR)
        jittor_img = jt.Var(np.array(img, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(0)
        jittor_img = jt.reshape(jittor_img,(3,256,256))
        return jittor_img,kpt,A
        
        
    
    
    def __getitem__(self,index:int):
        img,kpt,A = self.getImage(self.imgPaths[self.nowCategoty][index],self.kptPaths[self.nowCategoty][index])
        return jt.reshape(img,(1,3,256,256)),jt.reshape(kpt,(1,2,10)),jt.reshape(A,(1,10,10))
    
    def __iter__(self):
        index:int = 0
        while(index + self.batch_size*2<=self.nowLen):
            image1,kpt1,A1 = self[index]
            image2,kpt2,A2 = self[index+self.batch_size]
            for i in range(1,self.batch_size):
                img1_mid,kpt1_mid,A1_mid = self[(int)(index+i)]
                img2_mid,kpt2_mid,A2_mid = self[(int)(index+self.batch_size+i)]
                image1 = jt.concat((image1,img1_mid),dim=0)
                kpt1 = jt.concat((kpt1,kpt1_mid),dim=0)
                A1 = jt.concat((A1,A1_mid),dim=0)
                image2 = jt.concat((image2,img2_mid),dim=0)
                kpt2 = jt.concat((kpt2,kpt2_mid),dim=0)
                A2 = jt.concat((A2,A2_mid),dim=0)
            index = index+self.batch_size*2
            yield image1,kpt1,A1,image2,kpt2,A2
       
    def __len__(self):
        return self.nowLen

def testMyDataset():
    trainImgPaths,trainKptPaths,testImgPaths,testKptPaths = getPaths()
    trainDataset = MyDataset(trainImgPaths,trainKptPaths,batch_size=2)
    trainDataset.setCategory(0)
    for image1,kpt1,A1,image2,kpt2,A2 in trainDataset:
        print(image1.shape)
        print(kpt1.shape)
        print(A1.shape)
        print(image2.shape)
        break;


# 模型



def local_response_norm(input: Var, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0) -> Var:
    """
    jittor implementation of local_response_norm
    """
    dim = input.ndim
    assert dim >= 3

    if input.numel() == 0:
        return input

    div = input.multiply(input).unsqueeze(1)
    if dim == 3:
        div = nn.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = nn.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = nn.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = nn.AvgPool3d((size, 1, 1), stride=1)(div).squeeze(1)
        div = div.view(sizes)
        # print("size>=4")
        # print(dim)
    div = div.multiply(alpha).add(k).pow(beta)
    return input / div


def l2norm(node_feat):
    return local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)
    




class CNNNet(jt.nn.Module):
    def __init__(self, vgg16_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[:31]])
        self.edge_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[31:38]])

    def execute(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global

class GMNet(jt.nn.Module):
    def __init__(self,vgg16_cnn):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=False) # fetch the network object
        # self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain='voc') # fetch the network object
        self.cnn = CNNNet(vgg16_cnn)
         
        path = pygm.utils.download('vgg16_pca_voc_jittor.pt', 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1qLxjcVq7X3brylxRJvELCbtCzfuXQ24J')
        # path = pygm.utils.download('vgg16_cie_voc_jittor.pt', 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1wDbA-8sK4BNhA48z2c-Gtdd4AarRxfqT')
        self.cnn.load_state_dict(jt.load(path))
    
    def execute_old(self, img1, img2, kpts1, kpts2,A1,A2):
        # CNN feature extractor layers
        # A1 = delaunay_triangulation(kpts1)
        # A2 = delaunay_triangulation(kpts2)
        
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = jt.nn.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_global_upsample = jt.nn.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_local_upsample = jt.nn.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_global_upsample = jt.nn.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)

        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()
        node1 = feat1_upsample[0, :, rounded_kpts1[0,1], rounded_kpts1[0,0]].t()  # shape: NxC
        node2 = feat2_upsample[0, :, rounded_kpts2[0,1], rounded_kpts2[0,0]].t()  # shape: NxC

        # PCA-GM matching layers
        # X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        # X = pygm.ipca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        node1 = jt.reshape(node1,(1,node1.shape[0],node1.shape[1]))
        node2 = jt.reshape(node2,(1,node2.shape[0],node2.shape[1]))
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net)
        return X

    def execute(self, img1, img2, kpts1, kpts2,A1,A2):
        # CNN feature extractor layers
        
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = jt.nn.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_global_upsample = jt.nn.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_local_upsample = jt.nn.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_global_upsample = jt.nn.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)#batch,channel,n,n
        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()
       
        batch_size = feat1_upsample.shape[0]
        node1 = feat1_upsample[0, :, rounded_kpts1[0,1], rounded_kpts1[0,0]].t()  # shape: NxC
        N = node1.shape[0]
        C = node1.shape[1]
        # print(node1.shape)
        node1 = jt.reshape(node1,(1,N,C))
        node2 = feat2_upsample[0, :, rounded_kpts2[0,1], rounded_kpts2[0,0]].t()  # shape: NxC
        node2 = jt.reshape(node2,(1,N,C))
        # print(node1.shape)
        for index in range(1,batch_size):
            node1_mid = feat1_upsample[(int)(index), :, rounded_kpts1[(int)(index),1], rounded_kpts1[(int)(index),0]].t()
            node2_mid = feat2_upsample[(int)(index), :, rounded_kpts2[(int)(index),1], rounded_kpts2[(int)(index),0]].t()
            node1_mid = jt.reshape(node1_mid,(1,N,C))
            node2_mid = jt.reshape(node2_mid,(1,N,C))
           
            node1 = jt.concat((node1,node1_mid),dim=0)
            # print(node1.shape)
            node2 = jt.concat((node2,node2_mid),dim=0)
        
        # jt.sync_all(True)
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        # jt.sync_all(True)
        # X = pygm.pca_gm(node1, node2, A1, A2,  pretrain='voc') # the network object is reused
        
        return X
    


train_class:int = 5

class MyBenchmark():
    def __init__(self,train_dataset:MyDataset,test_dataset:MyDataset,model:GMNet,categories,optim):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.categories = categories
        self.categoriesNum = len(categories)
        self.optim = optim
        
    
    def train(self,epochNum):#梯度爆炸了
        print("----------train begin-------")
        bounds = [0,0,0,0,0]
        for epoch in range(epochNum):
            total_loss = []
            # loss_num = 0
            total_accu = []
            for categoryNumber in range(train_class):#self.categoriesNum
                if((bounds[(int)(categoryNumber)]>0.95) and (categoryNumber!=0) and (categoryNumber!=4)):
                    bounds[(int)(categoryNumber)] = 0
                    print("continue:"+self.categories[(int)(categoryNumber)])
                    continue;
                self.train_dataset.setCategory((int)(categoryNumber))
                # print(self.train_dataset.__len__())
                bound_accu = []
                for batch_idx,(jittor_img1,kpts1,A1,jittor_img2,kpts2,A2) in enumerate(self.train_dataset):
                    accu = 0.0
                    loss = jt.Var(0.0)
                    # jt.sync_all(True)
                    X = model(jittor_img1, jittor_img2, kpts1, kpts2,A1,A2)
                    # jt.sync_all(True)
                    for i in range(X.shape[0]):  
                        X_index = X[(int)(i)]
                        X_gt = jt.init.eye(X_index.shape[0])
                        loss += pygm.utils.permutation_loss(X_index, X_gt)
                        precision = (X_index * X_gt).sum() / X_index.sum()
                        accu+=((float)(precision))
                        
                    
                    loss = loss/self.train_dataset.batch_size    
                    accu = accu/self.train_dataset.batch_size
                    
                    bound_accu.append(accu)
                    
                    total_loss.append((float)(loss))
                    total_accu.append(accu)
                    
                    optim.backward(loss)
                    
                    # grad_size = []
                    # for param in self.model.parameters():
                    #     grad_size.append(jt.abs(param.opt_grad(optim)).mean().item())
                    # print(grad_size)
                    
                    optim.step()
                    optim.zero_grad()
                    # jt.sync_all()
                    # jt.gc()
                bounds[(int)(categoryNumber)] = np.mean(bound_accu)
                # print("bound=={:.6f} number=={}".format(bounds[(int)(categoryNumber)],categoryNumber)) 
                
                   
            print("------------this is epoch{}'s train loss=={:.6f} train accuracy=={:.6f}------------".format(epoch,np.mean(total_loss),np.mean(total_accu)))
            
        print("----------------train over-----------")       
        
        

    def test(self,epochNum):#梯度爆炸了
        print("----------test begin-------")
        for epoch in range(epochNum):
            accus = []
            total_accus = []
            for i in range(train_class):
                accus.append([])
            for categoryNumber in range(train_class):#self.categoriesNum
                self.test_dataset.setCategory((int)(categoryNumber))
                # print(self.test_dataset.__len__())
                for batch_idx,(jittor_img1,kpts1,A1,jittor_img2,kpts2,A2) in enumerate(self.test_dataset):
                    accu = 0.0
                    loss = jt.Var(0.0)
                    
                    X = model(jittor_img1, jittor_img2, kpts1, kpts2,A1,A2)
                    for i in range(X.shape[0]):  
                        X_index = X[(int)(i)]
                        X_gt = jt.init.eye(X_index.shape[0])
                        loss += pygm.utils.permutation_loss(X_index, X_gt)
                        precision = (X_index * X_gt).sum() / X_index.sum()
                        accu+=((float)(precision))
                        
                    
                    loss = loss/self.test_dataset.batch_size    
                    accu = accu/self.test_dataset.batch_size    
                    
                    # total_loss.append((float)(loss))
                    accus[(int)(categoryNumber)].append(accu)
                    total_accus.append(accu)
                   
                    
                
            # print(accus[1])
            # print(total_accus)      
            print("------------this is epoch{}'s\n car accuracy=={:.6f}\n duck accuracy accuracy=={:.6f}\n face accuracy accuracy=={:.6f}\n motorbike accuracy accuracy=={:.6f}\n winebottle accuracy accuracy=={:.6f}\n total accuracy accuracy=={:.6f}\n------------".
                  format(epoch,np.mean(accus[0]),np.mean(accus[1]),np.mean(accus[2]),np.mean(accus[3]),np.mean(accus[4]),np.mean(total_accus)))
            
        print("----------------test over-----------")   
        


trainImgPaths,trainKptPaths,testImgPaths,testKptPaths = getPaths()
trainDataset = MyDataset(trainImgPaths,trainKptPaths,batch_size=3)
testDataset = MyDataset(testImgPaths,testKptPaths,batch_size=3)



categories = ["Car","Duck","Face","Motorbike","Winebottle"]
vgg16_cnn = models.vgg16_bn(True)
trainEpoch = 10
testEpoch = 1
model = GMNet(vgg16_cnn)
optim = jt.optim.Adam(model.parameters(), lr=1e-4)
myBenchmark = MyBenchmark(trainDataset,testDataset,model,categories,optim)

# train
myBenchmark.train(trainEpoch)
# exit(0)
# test
myBenchmark.test(testEpoch)