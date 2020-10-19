# what-i-read
읽은 Paper들을 다음 3가지 관점에서 정리합니다. 1) 저자가 성취하려 한 것 2) 저자의 key approach 3) 뒤이어 읽을 것들  

## Contents

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

* Kyo-Joong et al., <Personalized news recommendation using classified keywords to capture user preference>, 16th International Conference on Advanced Communication Technology, 2014
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

  
