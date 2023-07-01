<div align="center">
  <h1>korcen</h1>
</div>

![131_20220604170616](https://user-images.githubusercontent.com/85154556/171998341-9a7439c8-122f-4a9f-beb6-0e0b3aad05ed.png)


키워드 기반의 검열 중 가장 뛰어난 성능을 보인 korcen이 딥러닝을 통해 더 강력해진 검열로 찾아왔습니다.

`직접 수집한 중복제외 200만개의 문장`과 `데이터셋을 딥러닝을 통해 분류 후` 이용하여 학습하였습니다.

해당 오픈소스는 **데모 버전**으로 최신 모델과 데이터 파일을 이용하시려면 문의해주세요

키워드 기반 기존 라이브러리 : [py version](https://github.com/KR-korcen/korcen),[ts version](https://github.com/KR-korcen/korcen.ts)

[서포트 디스코드 서버](https://discord.gg/wyTU3ZQBPE)

## 최신 모델이 적용된 프로젝트
>[TNS 봇](https://discord.com/api/oauth2/authorize?client_id=848795383751639080&permissions=8&scope=bot%20applications.commands)
```
Discord Bot
2000+ servers
```
## 예시 코드
>Python 3.10
```py
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 200

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
    sentence_seq = pad_sequences(tokenizer.texts_to_sequences([sentence]), maxlen=maxlen)
    prediction = model.predict(sentence_seq)[0][0]
    print(prediction)
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

### Github
- [KcBERT](https://github.com/Beomi/KcBERT/)
- [Toxic_comment_data](https://github.com/songys/Toxic_comment_data)

### Youtube
- [[NDC] 딥러닝으로 욕설 탐지하기](https://youtu.be/K4nU7yXy7R8)


# License
모든 `korcen`은 `Apache-2.0`라이선스 하에 공개되고 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 

- 라이선스 고지 및 저작권 고지 필수(일반인이 접근 가능한 부분에 표시)

Copyright© All rights reserved.
