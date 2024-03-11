<div align="center">
  <h1>Korcen</h1>
</div>

![131_20220604170616](https://user-images.githubusercontent.com/85154556/171998341-9a7439c8-122f-4a9f-beb6-0e0b3aad05ed.png)

korcen-ml은 기존 키워드 기반의 korcen의 우회가 쉽다는 단점을 극복하기위해 딥러닝을 통해 정확도를 한층 더 올리려는 프로젝트입니다.

KOGPT2 모델만 공개하고 있으며 모델 파일은 [여기](https://github.com/KR-korcen/korcen-ml/tree/main/model)에서 확인이 가능합니다.

더 많은 모델 파일과 학습 데이터를 다운받고 싶다면 문의주세요.

|  | 데이터 문장수 |
|------|------|
| VDCNN(23.4.30) | 200,000개 |
| VDCNN_KOGPT2(23.5.28) | 2,000,000개 |
| VDCNN_LLAMA2(23.9.30) | 5,000,000개 | 
| VDCNN_LLAMA2_V2(24.1.29) | 10,000,000개 |


키워드 기반 기존 라이브러리 : [py version](https://github.com/KR-korcen/korcen), [ts version](https://github.com/KR-korcen/korcen.ts)

[서포트 디스코드 서버](https://discord.gg/wyTU3ZQBPE)

## 모델 검증
데이터마다 욕설의 기준이 달라 오차가 있다는 걸 감안하고 확인하시기 바랍니다.


|  | [korean-malicious-comments-dataset](https://github.com/ZIZUN/korean-malicious-comments-dataset) | [Curse-detection-data](https://github.com/2runo/Curse-detection-data) | [kmhas_korean_hate_speech](https://huggingface.co/datasets/jeanlee/kmhas_korean_hate_speech) | [Korean Extremist Website Womad Hate Speech Data](https://www.kaggle.com/datasets/captainnemo9292/korean-extremist-website-womad-hate-speech-data/data) | [LGBT-targeted HateSpeech Comments Dataset (Korean)](https://www.kaggle.com/datasets/junbumlee/lgbt-hatespeech-comments-at-naver-news-korean) |
|------|------|------|------|------|------|
| [korcen](https://github.com/KR-korcen/korcen) | 0.7121 | **0.8415** | 0.6800 | 0.6305 | 0.4479 |
| VDCNN(23.4.30) | 0.6900 | 0.4885 |  | 0.4885 |  |
| VDCNN_KOGPT2(23.6.15) | 0.7545 | 0.7824 |  | 0.7055 | 0.6875 |
| VDCNN_LLAMA2(23.9.30) | 0.7762 | 0.8104 | 0.7296 |  |  |
| VDCNN_LLAMA2_V2(24.1.29) | **0.8322** | 0.8410 | **0.7837** | **0.7120** | **0.7477** |

## example
```py
#py: 3.10, tf: 2.10
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 1000

model_path = 'vdcnn_model.h5'
tokenizer_path = "tokenizer.pickle"

model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    
    return text

def predict_text(text):
    sentence = preprocess_text(text)
    encoded_sentence = tokenizer.encode_plus(sentence,
                                             max_length=maxlen,
                                             padding="max_length",
                                             truncation=True)['input_ids']
    sentence_seq = pad_sequences([encoded_sentence], maxlen=maxlen, truncating="post")
    prediction = model.predict(sentence_seq)[0][0]
    return prediction
    
while True:
    text = input("Enter the sentence you want to test: ")
    result = predict_text(text)
    if result >= 0.5:
        print("This sentence contains abusive language.")
    else:
        print("It's a normal sentence.")
```


## Maker


>Tanat
```
github:   Tanat05
discord:  Tanat05
email:    tanat@tanat.kr
```

## Reference


- [vilm/vulture-40B](https://huggingface.co/vilm/vulture-40b)
- [beomi/llama-2-ko-70b](https://huggingface.co/beomi/llama-2-ko-70b)
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
- [SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2)
- [Toxic_comment_data](https://github.com/songys/Toxic_comment_data)
- [[NDC] 딥러닝으로 욕설 탐지하기](https://youtu.be/K4nU7yXy7R8)
- [머신러닝 부적절 텍스트 분류:실전편](https://medium.com/watcha/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%B6%80%EC%A0%81%EC%A0%88-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EB%B6%84%EB%A5%98-%EC%8B%A4%EC%A0%84%ED%8E%B8-57587ecfae78)


# License
모든 `korcen`은 `Apache-2.0`라이선스 하에 공개되고 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 

- 라이선스 고지 및 저작권 고지 필수(일반인이 접근 가능한 부분에 표시)

Copyright© All rights reserved.
