># CNN으로 MNIST 분류하기.   
>convolution layer와 dense layer를 이용해서 MNIST 데이터 분류 모델을 학습시킨다.   
>MNIST데이터를 로딩하고, CNN 모델, test 데이터로 예측도 진행한다. 

<br><br>

>## 사전 준비   
>gpu, cpu 중 하나를 선택하고, 데이터를 불러온다. (colab에서 gpu로 실행)   
>DataLoader을 이용해서 데이터의 batch_size와 shuffle 속성을 정해준다.   


<br>

*필요한 패키지 불러오고, gpu 선택*   

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

%matplotlib inline
%config InlineBackend.figure_format ="retina"

print("PyTorch version:{%s}"%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device : {%s}"%(device))
```   

*파라미터 설정 및 데이터 로딩*   

```python
mnist_train = datasets.MNIST(root='/data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='/data/', train=False, transform=transforms.ToTensor(), download=True)

BATCH_SIZE = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = BATCH_SIZE, shuffle=True, num_workers=1)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = BATCH_SIZE, shuffle=True, num_workers=1)
```    

<br><br>


>## 모델 설정   
>CNN모델을 사용해서 분류를 학습시킬 것이기 때문에, 그에 맞는 모델 클래스를 설정한다.   


```python
class ConvolutionalNeuralNetworkClass(nn.Module):
  def __init__(self, name="cnn", xdim=[1,28,28], ksize=3, cdims=[32,64], hdims=[1024,128], ydim=10, USE_BATCHNORM=False):
    super(ConvolutionalNeuralNetworkClass, self).__init__()
    self.name = name
    self.xdim = xdim
    self.ksize = ksize
    self.cdims = cdims
    self.hdims = hdims
    self.ydim = ydim
    self.USE_BATCHNORM = USE_BATCHNORM   --> 기본 파라미터 설정


    # Convolutional layers      
    self.layers =[]  --> convolutional layer와 dense layer 담을 객체
    prev_cdim = self.xdim[0]    
    for cdim in self.cdims:
      self.layers.append(
      
      # 입력 데이터의 채널 수가 1이기 때문에, 그에 맞춰서 
      # convolutional layer의 dimension을 맞춘다.
      # 이 경우에는 1로 맞춘다. (반복하면서, 이전 dim을 따른다.)
      
          nn.Conv2d(in_channels=prev_cdim,
                    out_channels=cdim,
                    kernel_size=self.ksize,
                    stride=(1,1),
                    padding=self.ksize//2)
      )
      if self.USE_BATCHNORM:
        self.layers.append(nn.BatchNorm2d(cdim))
      self.layers.append(nn.ReLU(True))
      self.layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))  --> 커널 사이즈 2로 pool layer를 줬기 때문에, 사이즈가 1/2로 변함.
      self.layers.append(nn.Dropout2d(p=0.5))
      prev_cdim = cdim


    # Dense layers
    self.layers.append(nn.Flatten())   ---> 1차원으로 펴는 작업
    
    prev_hdim = prev_cdim*(self.xdim[1]//(2**len(self.cdims)))*(self.xdim[2]//(2**len(self.cdims)))
    for hdim in self.hdims:
      self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
      self.layers.append(nn.ReLU(True))
      prev_hdim = hdim
    self.layers.append(nn.Linear(prev_hdim, self.ydim, bias=True))


    # Concatenate all layers

    self.net = nn.Sequential()  ---> 지금껏 layers에 모은 layer들을 하나의 신경망으로 만든다.
    for l_idx, layer in enumerate(self.layers):
      layer_name = "%s_%02d"%(type(layer).__name__.lower(), l_idx)
      self.net.add_module(layer_name, layer)
    self.init_param()

  def init_param(self):   ---> 가중치 초기화
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
  def forward(self, x):  ---> forward 연산 실행.
    return self.net(x)
```

<br>

*입력값 shape 변화 과정*   

```
총 2개의 convolutional layer를 통해서, 

(28, 28, 1) --> (14, 14, 32) --> (7, 7, 64)

와 같은 tensor로 변하게 되고,
2개의 dense layer를 통해서, 

(7*7*64, 1024) --> (1024, 128) --> (10)

과 같은 벡터로 변하게 된다.
```   

<br><br>


>## 데이터 학습, 예측 (train, test)   
>CNN모델로 학습 데이터를 학습시킨 다음, 테스트 데이터를 이용해 예측을 진행한다.   

<br>

*train*   

```python
C.init_param() # 파라미터 생성.
C.train() # train모드로 변경
EPOCHS,print_every = 10,1  # 모델 학습 반복 수

for epoch in range(EPOCHS):  
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
    
        # Forward path
        y_pred = C.forward(batch_in.view(-1,1,28,28).to(device))
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        optm.zero_grad()      # reset gradient 
        loss_out.backward()      # backpropagate
        optm.step()      # optimizer update
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    
    # Print
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(C,train_iter,device)
        test_accr = func_eval(C,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr))
```   

<br>

*test*   
```python
n_sample = 25
sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
test_x = mnist_test.data[sample_indices]
test_y = mnist_test.targets[sample_indices]

with torch.no_grad():  ---> 이미 학습시킨 모델이므로, gradient를 추적하지 않는다.
    y_pred = C.forward(test_x.view(-1,1,28,28).type(torch.float).to(device))
y_pred = y_pred.argmax(axis=1)

```

>## 정리   
>train과 test 코드는 기본적으로 크게 달라지는 부분이 없다.   
>model 구성을 눈으로 익히고, 계속해서 연습하며, 
