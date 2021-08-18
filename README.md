# **ML Words And Notation For Beginners**

머신러닝을 처음 접할 때 만나는 무섭고 어려운 용어와 표기법들 때문에 다음과 같은 상황이 온다!😱 <br>
- *기존 공부의 흐름이 끊겨버림*
- *단어가 무엇이었는지 헷갈림*
- *검색해도 어려운 말이나 수식밖에 없음*

이것들을 대비하여 도움이 될 수 있도록 용어와 표기법을 간단하게만 정리하고 이해하고 공부하는 Repo! 🙌 <br>
(틀렸거나 / 추가하고싶거나 / 더 쉽게 설명할 수 있는 / 내용이 있다면 누구나 `PR` 이나 `Issue` 주세요!)
<br>

👉 **[용어항목으로](#🆎용어)** <br>
👉 **[표기법항목으로](#✍️표기법)** <br>

<br><br>



## **🆎용어**

- **Activation Function(활성화 함수)** 
    - 

- **Bias(편향)**
    - **`Neural Network(신경망)` 에서의 `Bias`** : 
        - 하나의 노드로 입력된 모든 값을 더한 뒤 그 값에 더해주는 **상수** 로, 노드의 활성화 함수를 거쳐 **최종적으로 출력되는 값을 조절하는(도와주는) 역할** 을 한다.  즉, 이 `상수` 를 더해줌으로써 활성화 함수의 그래프 적합도를 높이고, 모든 입력이 0인 경우에도 노드가 활성화되는 것을 보장하게 하여 편리함을 취하는 것이다. 상수는 보통 0으로 초기화되며 트레이닝을 통해 최적값을 가지게된다.

<br>

- **Cost Function(비용함수)**
    - 훈련세트들이 얼마나 잘 추측되었는지 측정해주는 함수. 
    - 일반적으로 `Cost Function(비용함수)` = `Loss Function(손실함수)` = `Object Function(목적 함수)` 모두 같은 말이라고 보는 편이다.

<br>

- **Data augmentation** : 너무 적은 데이터는 `Overfitting` 을 일으킬 수 있으니 사진과 같은 데이터를 좌우반전시키거나, 확대시키거나, 기울이는 등을 통해서 데이터를 확장시키는 것.

<br>

- **Early stopping** : 학습반복을 많이하면 `Overfitting` 이 일어날 수 있으니 중간에서 반복을 멈추는 방법. train 데이터의 `loss` 가 감소하고 검증 데이터의 `loss` 증가할 때 `Overfitting` 을 방지하기 위해 멈춘다. 

<br>

- **Loss Function(손실함수)**
    - 단일 훈련 셋에서 손실이 얼마나 발생했는지 측정해주는 함수. 
    - 일반적으로 `Loss Function(비용함수)` = `Cost Function(손실함수)` = `Object Function(목적 함수)` 모두 같은 말이라고 보는 편이다.

- **Maximum Likelihood Estimation(최대가능도)**

<br>

- **Normalization**
    - 머신러닝에서, 데이터가 가진 Feature의 범위가 너무 심하게 차이가 나서 영향력이 클 때 혼란스럽지않도록 범위를 0과 1 사이의 값으로 바꾸는 것.

<br>

- **Regularization**
    - 모델이 주어진 데이터에 너무 과하게 핏되어있을 때(`Overfitting(과적합)`),  변수의 양을 건드리는 대신에 값이 큰 특정 `weight(가중치)` 에 패널티를 주거나 노드를 Drop하는 등의 행위를 하여 예측함수를 일반화시켜주는 것.  
        - **L1 Regularizaiton(L1-norm)** : 기존 `Cost Funtion` 에 `weight` 들의 절대값을 더하여 일반화시키는 것.  `weight` 를 업데이트하기 위해서 이 `Cost Fucntion` 을 편미분하게 되면, 특정상수를 빼는 꼴이 되고 결국엔 작은 값을 가진 가중치는 weight가 0이 되어버린다. 이렇게 되면 영향을 크게 미치는 피쳐들만 남으면서 `Overfitting` 을 예방할 수 있다.
        - **L2 Regularizaiton(L2-norm)** : 기존 `Cost Funtion` 에 가중치의 절대값을 제곱한 것을 더하여 일반화시키는 것. `weight` 를 업데이트하기 위해서 이 `Cost Fucntion` 을 편미분하게 되면 `weight`의 값을 최대한 작게 만든다. 그렇게되면 전체적인 `weight` 의 값들이 작아지고 그에 따라 특정 `weight`가 널뛰지못하게되면 `Overfitting` 을 예방할 수 있다. 이렇게 `weight` 를 깎는 것을 `Weight Decay` 라고 부른다.
        - **Dropout** : 선택적으로 유닛을 Drop 하여 모델을 단순화 시켜 일반화 하는 것.
                

<br>

- **Residual(잔차)** 
    - **회귀식으로부터 얻은 예측값과 실제 관측값의 차이**를 뜻한다. 회귀식으로부터 그어진 일차방정식 직선과 실제 데이터의 수직 거리를 활용하여 잔차를 구한다. `Linear Regression(선형 회귀)`에서 이 `residual` 을 사용하여 `least squares(최소제곱)`방법을 이용한다. 당연하게도 이 `residual`이 적을수록 좋다.
        - **Error(오차)** - `Residual` 과 `Error` 는 다르다.
            - `Error` -  `Population(모집단)` 으로부터 추정한 회귀식과 실제 관측값의 차이.
            - `Residual` - `Sample(표본집단)` 로부터 추정한 회귀식과 실제 관측값의 차이.

<br>

- **Vector(벡터)**
    


- **Vectorization(벡터화)**
    - 일종의 병렬 처리로써 방대한 양의 데이터를 효율적으로 처리할 수 있게한다. 데이터의 양이 방대한 머신러닝에선 느리고 비효율적인 `Loop` 문을 사용하지않는 대신에 주로 이 `Vecotrization` 을 이용한다.

<br>

- **Weight(가중치)** 
    - 인공 신경망 모형의 하나인 퍼셉트론 알고리즘에서 나온 것으로 **노드과 노드간의 신호 혹은 연결의 세기(결과에 얼마나 영향을 미치는지)** 를 말한다. 이 `weight` 는 처음에는 랜덤하게 초기화되어있지만 점점 트레이닝 할수록 최적값을 가지게된다.(그래야만..한다....)
        - 처음 입력데이터가 주어지면 노드1은 입력받은 데이터에 랜덤하게 초기화된 `weight` 값을 곱한다.(결과에 끼치는 영향력을 조절한다.) 결과적으로 처음 입력된 데이터와는 다른 데이터를 노드2에게 건내주게된다.
        - 노드2는 마찬가지로 입력받은 데이터에 `weight` 를 곱하여 처음 입력받은 데이터와는 다른 데이터를 다음 노드에게 보낸다. 
        - 이를 반복하여 학습함으로써 최적의 `weight` 가 드러나게된다.

        이처럼 `weight`는 입력데이터를 바꿈으로써 결과값을 조절할 수 있기 때문에 **입력 데이터를 변환하는 신경망 내의 `Parameter(매게변수)`** 로 본다.


<br><br>

## **✍️표기법**

- **e (자연상수 e)** 
    - 𝝅(파이)가 3.14이듯이 `e` 는 2.71828을 나타내는 상수이다. 자연 성장에 대한 의미를 가지고 있어서 기울기등을 찾는 미적분학에 잘 어울린다. `Logistic function(로지스틱 함수)` 등에서 볼 수 있다.


<br>

- **$\vert\vert ⋅ \vert\vert$ norm(노름 or 놈)**
    - 벡터의 길이 혹은 크기를 측정하는 방법으로 norm으로 측정한 벡터의 크기 혹은 길이를 `Magnitude` 라고 한다. 간단히 말해서 벡터공간에서 거리를 구하는 함수.
    - 종류로는 `L1 Norm(Taxicab norm)`, `L2 Norm(Euclidean Norm)`, `L-infinity norm(상한 놈)` 등이 있다.
        - `L1 Norm(Taxicab norm)` : 단순 직선거리를 구하는 것이 아닌, 택시가 도시에서 길을 찾는 것과 같이 벡터요소의 절댓값의 핪을 구한다. 맨하탄 블록을 걸어다니는 것과 비슷하다고 해서 `Manhattan Distance` 라고도 한다. 
            - 만약 벡터 X가 [3, 4]라고 하면  $\vert\vert x \vert\vert$₁ = $\vert 3 \vert$ + $\vert 4 \vert$ 으로 계산하는 것이다.
            - ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/1874d7184b43bd33c09fc11bedeab479f3bedc42)
            - `L1 Regularization`, `Computer Vision` 에 쓰인다.
        - `L2 Norm(Euclidean Norm)` : 출발점에서부터 도착점까지의 가장 짧은 거리를 계산한다.
            - ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/50aa64b52c9ebd19d47552eae57ec4a05cf43e67)
            - `L2 Regularization`, `kNN 알고리즘`,  `kmean 알고리즘` 에 사용된다.
        - `L-infinity norm(상한 놈)` : 벡터의 각 요소 중에서 가장 큰 크기를 계산한다. `Maximum norm` 이라고도 한다.
            - ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/766d98a428ea0d3c6e96e75dae27721d810e9166)
    - `L1 Regularization`, `L2 Regularization` 등에 활용된다.
    - ![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Manhattan_distance.svg/300px-Manhattan_distance.svg.png)
        - *L1 Norm인 빨간색, 노란색, 파랑색 / L2 Norm은 초록색*

<br>

- **t** 
    - **`전치행렬 t(transposed matrix) `**
        - 행과 열을 교환하여 얻는 행렬. 머신러닝 계산을 할 때, 차원을 일치시키기위해 전치행렬을 자주 쓴다.
        - 전치행렬 과정 
        - ![https://ko.wikipedia.org/wiki/%EC%A0%84%EC%B9%98%ED%96%89%EB%A0%AC](https://upload.wikimedia.org/wikipedia/commons/e/e4/Matrix_transpose.gif)

<br>

- **ŷ (y_hat)** 
    - 예측결과값을 의미한다.
