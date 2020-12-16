## Contents
1. [Recommender System](#Recommender-System)

2. [Computer Vision](#Computer-Vision)

3. [Natural Language Processing](#Natural-Language-Processing)

4. [Etc](#Etc)



## Recommender System  
* Zhang et al., "NeuRec: On Nonlinear Transformation for Personalized Ranking", IJCAI, 2018
  + 저자가 성취한 것
  Latent factors에 Neural Network를 사용함으로서 Non-linearity를 확보하였고, 이는 기존의 Linear함을 전제로 한 MF보다 성능이 좋다
  + 저자의 Key Approach
  User Historical Data(Implicit Data - 직접적인 별점 아닌 클릭 같은 데이터)를 넣고 DBN 돌린 뒤 Items Embedding 가중치를 곱해서 User-based Neurec Prediction을 함.
     비슷한 논리로 Item-based NeuRec도 만듬.
  + 뒤이어 읽을 것
  논문에서는 Batch Normalization을 안 했고, 앞으로 해 볼 수도 있다고 했는데 내가 직접 해볼 수도 있음
  예전에 읽고 구현한 PMF랑 성능 차이가 얼마나 날지 직접 실험해보고 싶음  
  Evaluation Metrics 정리 잘 되어 있어서 나중에 플젝 할 때 참고

* Kyo-Joong et al., "Personalized news recommendation using classified keywords to capture user preference", 16th International Conference on Advanced Communication Technology, 2014
  + 저자가 성취한 것  
  News recommendation (Because it should be very quick, CF is not appropriate)
  + 저자의 Key Approach
  1. Using 5 features(Term freq, Inverted Term freq, Title, First Sentence, Cumultated Preference Weight) to make user’s profile(keyword and score
  2. Scoring news with same logic
  3. News ranking
  + 뒤이어 읽을 것들
  DBN 썼는데 비슷한 데이터셋으로 더 좋은 architecture를 쓴 연구자료를 찾아봐야 될 듯  
  
* Sedhain et al., "AutoRec: Autoencoders Meet Collaborative Filtering", WWW, 2015
  + 저자가 성취한 것  
  AutoEncoder를 Collaborative Filtering에 적용
  + 저자의 Key Approach
  AutoEncoder로 Latent Layer를 구현함. V와 W는 Fully connected. Item-based AutoRec 구현함.
  + 뒤이어 읽을 것
  AutoEncoder는 MF랑 논리가 되게 비슷해서 성능도 좋을 것 같음. AutoEncoder 자체에 대한 공부 + 추천시스템이 AutoEncoder 적용한 2019년 이후 논문 찾아볼 것

* Vinh Vo et al., "Generation Meets Recommendation: Proposing Novel Items for Group of Users", RecSyS, 2018
  + 저자가 성취한 것
  Variational AutoEncoder와 기존 유저들의 Embedding을 활용해서 Novel Item Embedding을 생성해 냄. 있는 Embedding을 예측하는 것이 아니라 있음직한 Embedding을 생성해낸 것이 키포인트. 
  + 저자의 Key Approach
  VAE를 이용함. Embedding space Z 위에서 Encoder와 Decoder를 학습시키는데, 생성한 embedding vector가 specifit rating function 연산의 최댓값을 만족하도록 training  
  + 뒤이어 읽을 것  
  
* Loepp et al., "Impact of Item Consumption on Assessment of Recommendations in User Studies", RecSyS, 2018
  + 저자가 성취한 것
  실제 추천 시스템에 의해 추천된 물건이 소비되었을 때 사용자에게 미치는 영향에 대해 Movie/Music case로 나누어 탐구
  + 저자의 Key Approach
  영화와 음악으로 나누어서 살펴보았는데, 음악의 경우 추천시스템에 의한 선곡이 실제로 재생될 수 있는 경우와 없는 경우의 별점 상관관계를 보았고, 영화 역시 보고난 후와 뒤의 상관관계를 시간을 두고 살펴봄
  + 뒤이어 읽을 것  
  
## Computer Vision  
* He et al., "Deep Residual Learning for Image Recognition", CVPR, 2016
  + 저자가 성취한 것
  Outperforming ImageNet classification (Won ILSVRC 2015 1st prize)
  + 저자의 Key Approach
  Deep residual learning framework 
  Identity Mapping by shortcuts  
  1x1 conv for reducing computational costs (Bottleneck)  
  + 뒤이어 읽을 것  
  
* Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR, 2016
  + 저자가 성취한 것  
  Classification + Sliding Window의 느린 속도를 극복하고 Object Detection을 Single regression problem with convnet로 바꿈
  + 저자의 Key Approach  
  이미지를 그리드로 나누어서, 그리드마다 (B * 5 + C)의 텐서를 부여함 (B : anchor box, C : 분류하고자 하는 클래스)  
  탁월한 loss function(큰 박스와 작은 박스를 위한 root on w/h  
  Non-max supression
  + 뒤이어 읽을 것  
  연구장학생 프로젝트 때 YOLOv2를 다뤘었는데, 다시 읽으니까 Loss Function이 정말 아름답다. YOLOv4 아직 못 읽어봤는데 읽어야 된다.
  
* Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", 2017
  + 저자가 성취한 것  
  컴퓨팅 파워가 좋지 않은 기기에서도 돌아갈 수 있도록 ConvNet의 연산량을 줄이면서 성능은 유지  
  + 저자의 Key Approach  
  Depthwise Convolution + Pointwise Convolution  
  + 뒤이어 읽을 것  
  모델의 경량화에 대한 후속 
  
* Chollet, "Xception: Deep Learning with Depthwise Seperable Convolutions", CVPR, 2017
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
  
* Girshick et al., "Rich feature hierarchies for accurate object detection and semantic segmantaion", CVPR, 2014
  + 저자가 성취한 것  
  2-stage-detector의 서막. R-CNN(Regions with CNN features)  
  + 저자의 Key Approach  
  전처리 된 이미지로부터 proposal region을 추출하는 것이 첫 번째 CNN. 이 영역들의 feature들을 추출하는 것이 두 번째 CNN. 이렇게 나온 feature들을 SVM으로 classification 
  + 뒤이어 읽을 것  
  Faster-R-CNN에서 이 두 개의 CNN을 어떻게 하나로 통합하면서도 2-stage-detector의 architecture를 유지하는지 다시 검토   
  
* Lin et al., "Focal Loss for Dense Object Detection", ICCV, 2017
  + 저자가 성취한 것  
  1-stage-detector로 2-stage-detector 수준의 성능을 얻기 위해 Class Imbalance 문제를 해결한 것
  + 저자의 Key Approach  
  Loss function을 Focal Loss라는 개념으로 보정하면서, correctly classified situation에서 loss값을 줄여주었음  
  + 뒤이어 읽을 것  
  다른 논문에서도 Feature Pyramid Network(FPN) 개념이 계속 등장하던데, 이거 따로 읽은 다음에 다시 읽어봐야 됨     
  
 * Dong et al., "Accelerating the Super-Resolution Convolutional Neural Network", ECCV, 2016
  + 저자가 성취한 것  
  기존의 SRCNN보다 더 빠른 FSRCNN을 제안
  + 저자의 Key Approach  
  1) Interpolation을 하지 않고 원래의 Image를 바로 사용.  
  2) HR Feature를 여러 Layer로 쪼갬
  + 뒤이어 읽을 것  
  
## Natural Language Processing
* Micolov et al., "Distributed Representations of Words and Phrases and their Compositionality", NIPS, 2013
  + 저자가 성취한 것  
  이전에 저자가 발표했던 Skip-gram model을 개선해서 더 성능 좋은 Word/Phrase Embedding을 구현함  
  + 저자의 Key Approach  
  1) Negative Sampling : target word 1개와 k개의 negative sample로 hierachical softmax를 대체함. Softmax의 분모에 있던 연산량 많은 항을 n개의 binary classification으로 변환해서 연산량을 줄임  
  2) Subsampling of Frequent words : the, a, in 같은 frequent word와 rare word의 비대칭을 보정하기 위한 subsampling approach  
  3) Phrase Skip-Gram : word vector 사이의 linear한 연산이 유의미하다는 것을 이용한 linear structure  
  + 뒤이어 읽을 것  
  GloVe 

* Pennington et al., "GloVe: Global Vectors for Word Representations", 2014
  + 저자가 성취한 것  
  기존 word embedding의 두 축인 LSA(Latenet Semantic Analysis)와 Shallow Window-based methods(Word2Vec)을 넘어선 Word embedding method    
  + 저자의 Key Approach  
  word vector의 내적과 co-occurence matrix를 모두 활용한 loss function 설정 (유도과정 다시 읽어봐야 됨 다 이해 못함)  
  + 뒤이어 읽을 것  

* Bahdanau et al., "Neural Machine Translation By Jointly Learning To Align And Translate", ICLR, 2015
  + 저자가 성취한 것  
  기존의 Seq2Seq model의 문제들(Vanishing gradients, 모든 문장을 fixed-length vector로 표현하다보니 긴 문장에서 성능 나빠지는 점)을 개선하기 위해 새로운 모델 아키텍쳐를 도입  
  + 저자의 Key Approach  
  Alignment model with Context Vector : Softmax의 결과값으로 표현되는 BRNN Encoder의 weighted sum을 Y를 선택하는 데 사용하는 아이디어.  
  이렇게 하면 decoder 단에서 source sentence의 어떤 부분에 pay attention을 해야 하는지 계산되기 때문에 fixed input vector를 넣을 필요가 없어짐
  + 뒤이어 읽을 것  
  Attention is all you need, BERT, Transformer 최신 연구 동향 등. 
  
* Vaswani et al., "Attention Is All You Need" , NIPS, 2017
  + 저자가 성취한 것  
  혁명적인 시도. Recurrent Network 구조를 버리고 Attention만으로 Transformer라는 새로운 아키텍처를 만들어서 Machine Translation에서 뛰어난 성능을 보임
  + 저자의 Key Approach  
  1) Scaled Dot-Product attention : Product attention은 원래 Attention이랑 같은데 1/루트 dk 항을 넣어서 dot product가 커질 때 gradient 크기 작아지는 것을 보정  
  2) Multi-Head Attention  
  Attention을 한번만 하는 게 아니라 V, K, Q를 Linear Layer에 넣고 여러 값으로 바꾼 뒤에 h번 진행 후 concatenation  
  3) Masked Multi-Head Attention  
  이전의 결과값에 대한 Attention을 취하기 위해 마스크 행렬 윗단을 -inf로 세팅하고 곱해주는 작업
  + 뒤이어 읽을 것  
  Transformer를 개선한 예시, Transformer와 Recommender System의 결합의 예

* Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2019. 
  + 저자가 성취한 것  
  Masked LM과 Next Sentence Prediction(NSP)를 활용한 Bidirectional Transformer로 Fine-Tuning Based Language Model를 만듬.  
  + 저자의 Key Approach  
  1) Masked LM : WordPiece 기반의 Embedding으로 15%는 MASK Embedding. 80%는 [MASK] Token, 10%는 Unchanged, 10%는 Random
  2) NSP : 두 개의 문장을 이어서 동시에 트레이닝. 절반은 연관 있고 절반은 연관 없음.  
  3) Fine-Tuning Approach : General한 Language Model. 여러 가지의 Task를 윗단의 Layer만 튜닝해서 사용할 수 있도록 Pre-Training / Fine-Tuning 단계를 나눔
  + 뒤이어 읽을 것  
  GPT-3  

## Etc
* Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", NeurIPS,  2018. 
  + 저자가 성취한 것  
  기존의 GBM, XGBoost가 가지고 있던 Prediction Shift 문제를 극복하여 더 빠르고 더 성능좋은 Boosting 알고리즘을 제시함.  
  + 저자의 Key Approach  
  1) Categorical Features : Permuation 기반의 Ordered Target Statistics(TS)
  2) Ordered Boosting : Permutation 기반의 split으로 이전의 데이터만으로 boosting 진행
  + 뒤이어 읽을 것  
  PyData 2018 London에서의 CatBoost 튜토리얼 영상 보고 실습하기      
