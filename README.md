## Contents
1. [Recommender System](#Recommender-System)

2. [Computer Vision](#Computer-Vision)

3. [Natural Language Processing](#Natural-Language-Processing)

4. [Etc](#Etc)



## Recommender System  
* Zhang et al., <NeuRec: On Nonlinear Transformation for Personalized Ranking>, IJCAI, 2018
  + 저자가 성취한 것
  Latent factors에 Neural Network를 사용함으로서 Non-linearity를 확보하였고, 이는 기존의 Linear함을 전제로 한 MF보다 성능이 좋다
  + 저자의 Key Approach
  User Historical Data(Implicit Data - 직접적인 별점 아닌 클릭 같은 데이터)를 넣고 DBN 돌린 뒤 Items Embedding 가중치를 곱해서 User-based Neurec Prediction을 함.
     비슷한 논리로 Item-based NeuRec도 만듬.
  + 뒤이어 읽을 것
  논문에서는 Batch Normalization을 안 했고, 앞으로 해 볼 수도 있다고 했는데 내가 직접 해볼 수도 있음
  예전에 읽고 구현한 PMF랑 성능 차이가 얼마나 날지 직접 실험해보고 싶음  
  Evaluation Metrics 정리 잘 되어 있어서 나중에 플젝 할 때 참고

* Kyo-Joong et al., Personalized news recommendation using classified keywords to capture user preference, 16th International Conference on Advanced Communication Technology, 2014
  + 저자가 성취한 것  
  News recommendation (Because it should be very quick, CF is not appropriate)
  + 저자의 Key Approach
  1. Using 5 features(Term freq, Inverted Term freq, Title, First Sentence, Cumultated Preference Weight) to make user’s profile(keyword and score
  2. Scoring news with same logic
  3. News ranking
  + 뒤이어 읽을 것들
  DBN 썼는데 비슷한 데이터셋으로 더 좋은 architecture를 쓴 연구자료를 찾아봐야 될 듯  
  
* Sedhain et al., <AutoRec: Autoencoders Meet Collaborative Filtering>, WWW, 2015
  + 저자가 성취한 것  
  AutoEncoder를 Collaborative Filtering에 적용
  + 저자의 Key Approach
  AutoEncoder로 Latent Layer를 구현함. V와 W는 Fully connected. Item-based AutoRec 구현함.
  + 뒤이어 읽을 것
  AutoEncoder는 MF랑 논리가 되게 비슷해서 성능도 좋을 것 같음. AutoEncoder 자체에 대한 공부 + 추천시스템이 AutoEncoder 적용한 2019년 이후 논문 찾아볼 것

  
## Computer Vision  
* He et al., <Deep Residual Learning for Image Recognition, CVPR, 2016
  + 저자가 성취한 것
  Outperforming ImageNet classification (Won ILSVRC 2015 1st prize)
  + 저자의 Key Approach
  Deep residual learning framework 
  Identity Mapping by shortcuts  
  1x1 conv for reducing computational costs (Bottleneck)  
  + 뒤이어 읽을 것  
  
* Redmon et al., <You Only Look Once: Unified, Real-Time Object Detection>, CVPR, 2016
  + 저자가 성취한 것  
  Classification + Sliding Window의 느린 속도를 극복하고 Object Detection을 Single regression problem with convnet로 바꿈
  + 저자의 Key Approach  
  이미지를 그리드로 나누어서, 그리드마다 (B * 5 + C)의 텐서를 부여함 (B : anchor box, C : 분류하고자 하는 클래스)  
  탁월한 loss function(큰 박스와 작은 박스를 위한 root on w/h  
  Non-max supression
  + 뒤이어 읽을 것  
  연구장학생 프로젝트 때 YOLOv2를 다뤘었는데, 다시 읽으니까 Loss Function이 정말 아름답다. YOLOv4 아직 못 읽어봤는데 읽어야 된다.
  
* Howard et al., <MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications>, 2017
  + 저자가 성취한 것  
  컴퓨팅 파워가 좋지 않은 기기에서도 돌아갈 수 있도록 ConvNet의 연산량을 줄이면서 성능은 유지  
  + 저자의 Key Approach  
  Depthwise Convolution + Pointwise Convolution  
  + 뒤이어 읽을 것  
  모델의 경량화에 대한 후속 
  
* Chollet, <Xception: Deep Learning with Depthwise Seperable Convolutions>, CVPR, 2017
  + 저자가 성취한 것  
  Cross-channel correlation과 Spatial correlation을 분리한 Inception을 응용해서 Xception architecture를 만듬  
  + 저자의 Key Approach  
  1x1 Conv로 Cross-channel correlation을 나타내는 channel들을 만든 뒤, 모든 channel을 분리시켜서 3x3 conv에 넣음  
  => Cross-channel과 Spatial을 완벽하게 분리시킬 수 있다!  
  => 이 때 이러한 방식이 기존의 depthwise convolution과 다른 건,  
  1) 1x1 conv 먼저 해서 cross-channel 연산 먼저 함 (vs spatial 3x3 conv 먼저 하고 1x1 conv하는 depthwise conv)    
  2) 1x1 conv 하고 ReLU에 넣은 다음 3x3 conv를 함 (vs conv 사이에 activation 없음)  
  + 뒤이어 읽을 것  
  Open source implementation  
  
## Natural Language Processing
* Micolov et al., <Distributed Representations of Words and Phrases and their COmpositionality, NIPS, 2013
  + 저자가 성취한 것  
  이전에 저자가 발표했던 Skip-gram model을 개선해서 더 성능 좋은 Word/Phrase Embedding을 구현함  
  + 저자의 Key Approach  
  1) Negative Sampling : target word 1개와 k개의 negative sample로 hierachical softmax를 대체함. Softmax의 분모에 있던 연산량 많은 항을 n개의 binary classification으로 변환해서 연산량을 줄임  
  2) Subsampling of Frequent words : the, a, in 같은 frequent word와 rare word의 비대칭을 보정하기 위한 subsampling approach  
  3) Phrase Skip-Gram : word vector 사이의 linear한 연산이 유의미하다는 것을 이용한 linear structure  
  + 뒤이어 읽을 것  
  GloVe 

* Pennington et al., <GloVe: Global Vectors for Word Representations>, 2014
  + 저자가 성취한 것  
  기존 word embedding의 두 축인 LSA(Latenet Semantic Analysis)와 Shallow Window-based methods(Word2Vec)을 넘어선 Word embedding method    
  + 저자의 Key Approach  
  word vector의 내적과 co-occurence matrix를 모두 활용한 loss function 설정 (유도과정 다시 읽어봐야 됨 다 이해 못함)  
  + 뒤이어 읽을 것  





## Etc
