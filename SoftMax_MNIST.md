# MNIST 데이터 분류   

*****   

<br>

*colab에서 코드 실행*   


<br>

>## GPU 설정   

<br>
 
*colab에서 노트 설정을 gpu로 설정했음.*   
```
USE_CUDA = torch.cuda.is_available() ---> GPU를 사용가능하면 True, 아니라면 False를 리턴

device = torch.device("cuda" if USE_CUDA else "cpu") ---> GPU 사용 가능하면 사용 아니면 CPU 사용
```    

<br><br>

>## seed값 설정 및 데이터 불러오기   

<br>

*torchvision.datasets을 이용해서 MNIST 데이터 불러오기*   

```python
#seed 값 설정
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)


#epoch, batch_size 설정
training_epochs = 15
batch_size = 100


#데이터 불러오기
mnist_train = dsets.MNIST(root='MNIST_data/',   
                          train=True,  ---> True 면 train 데이터셋 , False면 test 데이터셋
                          transform=transforms.ToTensor(),  ---> tensor로 변환
                          download=True)  ---> 데이터가 있으면 불러오고, 없다면 다운로드 하고 불러온다.

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

```   

*dataloader를 통해 데이터 셋 설정*      
*drop_last는 한 iteration이 batch_size 미달일 시, 버린다.*   
```python
data_loader = DataLoader(dataset=mnist_train, batch_size = batch_size,
                         shuffle=True, drop_last = True)  
```   

<br><br>

>## 모델 설계   

<br>

*nn.Linear를 활용 (high level 구현)*    

```python
linear = nn.Linear(784, 10, bias=True).to(device)

criterion = nn.CrossEntropyLoss().to(device)   ---> 내부적으로 소프트맥스 함수 포함.
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.1)  ---> 확률적 경사하강법으로 optimizer 설정.   
```      

<br>


>## 학습 및 테스트   

<br>

*epochs만큼 학습시키며, 매 학습마다 optimizer update*   

```python
for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = len(data_loader)

  for X, Y in data_loader:  # X = data , Y = label
    X = X.view(-1, 28*28).to(device)     ---> reshape
    Y = Y.to(device)

    optimizer.zero_grad() 
    
    hypothesis = linear(X)  --->  분류결과 (예측)
    
    cost = criterion(hypothesis, Y) --->  cross_entropy 계산 (cost)
    
    cost.backward()   ---> 역전파 계산(미분)
    optimizer.step()  ---> optimizer update
    
    avg_cost += cost / total_batch
```   

<br>

*임의의 테스트 데이터로 정확도 평가*   

```python
with torch.no_grad():  --->  torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.

    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
  

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    single_prediction = linear(X_single_data)
    
```



