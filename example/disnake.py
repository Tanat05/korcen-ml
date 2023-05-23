#disnake 
import asyncio
import disnake
from disnake.ext import commands, tasks
import tensorflow as tf
import numpy as np
import pickle
from konlpy.tag import Okt
from hanspell import spell_checker
import re
from korcen import korcen
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences

intents = disnake.Intents().all()
bot = commands.Bot(intents=intents, owner_id=922804740432203786)

maxlen = 200

model_path = 'vdcnn_model.h5'

tokenizer_path = "tokenizer.pickle"

# 모델과 토크나이저 로드
model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# 문장이 욕설인지 아닌지 판별하는 함수
async def predict_text(text):
    text = str(text).lower()

    # 디스코드 채널 형식 (@channel_name)
    text = re.sub(r'@[\w]+', '', text)
    
    # 디스코드 사용자 형식 (@username)
    text = re.sub(r'@[\S]+', '', text)
    
    # 이모지 형식 (:emoji_name:)
    text = re.sub(r':[a-zA-Z0-9_]+:', '', text)
    
    # 링크 형식 (http 또는 https로 시작하는 문자열)
    text = re.sub(r'http[s]?://\S+', '', text)

    sentence = text.strip()

    sentence_seq = pad_sequences(tokenizer.texts_to_sequences([sentence]), maxlen=maxlen)
    prediction = model.predict(sentence_seq)[0][0]
    print(prediction)
    return prediction

@bot.event
async def on_ready():
    print(f"[!] 다음으로 로그인에 성공했습니다.")
    print(f"[!] 다음 : {bot.user.name}")
    print(f"[!] 다음 : {bot.user.id}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    start = time.time()
    #text = spell_checker.check(message.content)
    #text.as_dict()
    #text.checked
    text = message.content
    result = await predict_text(text)
    if result >= 0.5:
        await message.delete()
        await message.channel.send(f"\"{text}\" 메세지를 삭제했습니다\n{int(time.time()-start)}초 소요")





bot.run("token")
