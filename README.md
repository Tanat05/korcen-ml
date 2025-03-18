<div align="center">
  <h1>Korcen-ML</h1>
</div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/85154556/171998341-9a7439c8-122f-4a9f-beb6-0e0b3aad05ed.png" alt="131_20220604170616">
</p>

<p align="center">
  korcen-ml은 기존 키워드 기반 korcen의 우회가 쉽다는 단점을 극복하기 위해<br>
  딥러닝을 통해 정확도를 향상시키려는 프로젝트입니다.
</p>

<p align="center">
  현재 KoGPT2 모델만 공개하고 있으며, 모델 파일은 <a href="https://github.com/KR-korcen/korcen-ml/tree/main/model">여기</a>에서 확인 가능합니다.<br>
  더 많은 모델 파일과 학습 데이터를 원하시면 문의해주세요.
</p>

<div align="center">

| 모델                    | 데이터 문장 수 |
| :---------------------- | -------------: |
| VDCNN (23.4.30)        |        200,000 |
| [VDCNN_KOGPT2](https://github.com/KR-korcen/korcen-ml/tree/main/model) (23.06.15) |     2,000,000 |
| VDCNN_LLAMA2 (23.09.30) |     5,000,000 |
| VDCNN_LLAMA2_V2 (24.06.04) |   13,000,000 |
| LSTM_EXAONE3 (24.08.16) |   13,000,000 |

</div>

<br>

키워드 기반 기존 라이브러리:  [py version](https://github.com/KR-korcen/korcen), [ts version](https://github.com/KR-korcen/korcen.ts)

[서포트 디스코드 서버](https://discord.gg/wyTU3ZQBPE)

## 모델 검증

<p>
  데이터마다 욕설의 기준이 달라 오차가 있을 수 있음을 감안하고 확인하시기 바랍니다.
</p>

<div align="center">

| 모델                                                       | [korean-malicious-comments-dataset](https://github.com/ZIZUN/korean-malicious-comments-dataset) | [Curse-detection-data](https://github.com/2runo/Curse-detection-data) | [kmhas_korean_hate_speech](https://huggingface.co/datasets/jeanlee/kmhas_korean_hate_speech) | [Korean Extremist Website Womad Hate Speech Data](https://www.kaggle.com/datasets/captainnemo9292/korean-extremist-website-womad-hate-speech-data/data) | [LGBT-targeted HateSpeech Comments Dataset (Korean)](https://www.kaggle.com/datasets/junbumlee/lgbt-hatespeech-comments-at-naver-news-korean) |
| :--------------------------------------------------------- | ----------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------: |
| [korcen](https://github.com/KR-korcen/korcen)             |                                                                                             0.7121 |                                                                                              0.8415 |                                                                                                   0.6800 |                                                                                                                                   0.6305 |                                                                                                                               0.4479 |
| TF VDCNN (23.4.30)                                         |                                                                                             0.6900 |                                                                                              0.4885 |                                                                                                          |                                                                                                                                   0.4885 |                                                                                                                                      |
| TF [VDCNN_KOGPT2](https://github.com/KR-korcen/korcen-ml/tree/main/model) (23.06.15) |                                                                                             0.7545 |                                                                                              0.7824 |                                                                                                          |                                                                                                                                   0.7055 |                                                                                                                               0.6875 |
| TF VDCNN_LLAMA2 (23.09.30)                                 |                                                                                             0.7762 |                                                                                              0.8104 |                                                                                                   0.7296 |                                                                                                                                          |                                                                                                                                      |
| TF VDCNN_LLAMA2_V2 (24.06.04)                               |                                                                                   **0.8322** |                                                                                              0.8420 |                                                                                                   0.7837 |                                                                                                                                   0.7120 |                                                                                                                         **0.7477** |
| TF LSTM_EXAONE3 (24.08.16)                                 |                                                                                   **0.8395** |                                                                                           **0.8432** |                                                                                                **0.8851** |                                                                                                                                **0.7155** |                                                                                                                               0.6919 |
</div>
## 예제 (Example)

```python
# py: 3.10, tf: 2.10
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 1000

model_path = 'vdcnn_model.h5'  # 모델 파일 경로
tokenizer_path = "tokenizer.pickle"  # 토크나이저 파일 경로

# 모델 로드
model = tf.keras.models.load_model(model_path)

# 토크나이저 로드
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()  # 소문자 변환
    return text

def predict_text(text):
    sentence = preprocess_text(text)
    # 토크나이저를 사용하여 문장을 인코딩
    encoded_sentence = tokenizer.encode_plus(
        sentence,
        max_length=maxlen,
        padding="max_length",  # 최대 길이에 맞게 패딩
        truncation=True  # 최대 길이 초과 시 잘라냄
    )['input_ids']

    sentence_seq = pad_sequences([encoded_sentence], maxlen=maxlen, truncating="post")
    prediction = model.predict(sentence_seq)[0][0]  # 예측 수행
    return prediction

# 사용자 입력 및 결과 출력
while True:
    text = input("Enter the sentence you want to test: ")  # 테스트할 문장 입력
    result = predict_text(text)
    if result >= 0.5:
        print("This sentence contains abusive language.")  # 욕설 포함
    else:
        print("It's a normal sentence.")  # 정상 문장
```
## 참고 자료 (Reference)
- vilm/vulture-40B
- beomi/llama-2-ko-70b
```
@misc {l._junbum_2023,
    author       = { {L. Junbum} },
    title        = { llama-2-ko-70b },
    year         = 2023,
    url          = { https://huggingface.co/beomi/llama-2-ko-70b },
    doi          = { 10.57967/hf/1130 },
    publisher    = { Hugging Face }
}
```
- LG AI Research/EXAONE-v3.0-Instruct-7.8B
```
@article{exaone-3.0-7.8B-instruct,
  title={EXAONE 3.0 7.8B Instruction Tuned Language Model},
  author={LG AI Research},
  journal={arXiv preprint arXiv:2408.03541},
  year={2024}
}
```
- [SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2)
- [Toxic_comment_data](https://github.com/songys/Toxic_comment_data)
- [[NDC] 딥러닝으로 욕설 탐지하기](https://youtu.be/K4nU7yXy7R8)
- [머신러닝 부적절 텍스트 분류:실전편](https://medium.com/watcha/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%B6%80%EC%A0%81%EC%A0%88-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EB%B6%84%EB%A5%98-%EC%8B%A4%EC%A0%84%ED%8E%B8-57587ecfae78)

## 라이선스 (License)
모든 korcen은 Apache-2.0 라이선스 하에 공개됩니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요.

- 라이선스 고지 및 저작권 고지 필수 (일반인이 접근 가능한 부분에 표시)

Copyright © All rights reserved.
