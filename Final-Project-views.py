# myproject/myapp/views.py

from django.shortcuts import render
from teanaps.text_analysis import SentimentAnalysis
from teanaps.text_analysis import DocumentSummarizer
from django.http import JsonResponse
import sqlite3
import pandas as pd
import numpy as np
from teanaps.nlp import Processing
pro = Processing()
from gensim.models import Word2Vec
import sqlite3
from bs4 import BeautifulSoup
import re
# from .models import Example, IncheonSeogu  # 모델 클래스를 임포트


def text_analysis(request):
    if request.method == 'POST':

        # 텍스트 추출
        search_word = request.POST.get('search_word', '')
        
        # DB를 통해서 df 불러오기
        df_search = load_text(search_word)

        # 본문기사 변수로 지정
        text1 = df_search[0][2]
        text2 = df_search[1][2]

        # 감성분석
        senti = SentimentAnalysis()
        result_sen1 = senti.tag(text1, neutral_th=0.5)
        result_sen2 = senti.tag(text2, neutral_th=0.5)

        # 본문 요약, LSA 사용(생성요약)
        ds = DocumentSummarizer()
        SUMMARIZER_TYPE = "lsa"
        MAX_SENTENCES = 3
        result_sum1 = ds.summarize(SUMMARIZER_TYPE, MAX_SENTENCES, document= text1)
        result_sum2 = ds.summarize(SUMMARIZER_TYPE, MAX_SENTENCES, document= text2)

        context = {             
            # 유사도 1등 기사
            'title1' : f"{df_search[0][0]}",
            'date1' : f"{df_search[0][1]}",
            'result_sen1': f"{result_sen1[0]}, '{result_sen1[1]}'",
            'result_sum1': ' '.join([f'{word}' for word in result_sum1]),
            'link1' : f"{df_search[0][3]}",

            # 유사도 2등 기사
            'title2' : f"{df_search[1][0]}",
            'date2' : f"{df_search[1][1]}",
            'result_sen2': f"{result_sen2[0]}, '{result_sen2[1]}'",
            'result_sum2': ' '.join([f'{word}' for word in result_sum2]),
            'link2' : f"{df_search[1][3]}",
        }

        return JsonResponse(context)

    return render(request, 'text_analysis.html')





def load_text(search_word):
    
    # SQLite 데이터베이스에 연결
    conn = sqlite3.connect('embedding_database.db')
    # 테이블 조회 쿼리
    query = "SELECT * FROM embedding_table"
    # 쿼리 실행하여 데이터프레임으로 저장
    df = pd.read_sql_query(query, conn)
    # 연결 종료
    conn.close()


    # 한글 텍스트로 Word2Vec 모델 학습
    corpus = [sentence.split() for sentence in df['Content']]
    w2v_model = Word2Vec(corpus, size=100, window=5, min_count=1, sg=0)

    # 텍스트를 임베딩하여 벡터로 변환하는 함수
    def embed_text(text):
        words = text.split()
        vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if vectors:
            return sum(vectors) / len(vectors)
        else:
            return None
        
    # DataFrame의 '한글텍스트' 열을 임베딩하여 'Embedding' 열 생성
    df['Embedding'] = df['Content'].apply(embed_text)

    # 검색어 임베딩 구하기
    target_vector = embed_text(search_word).astype(float)

    # 코사인 유사도 계산 함수
    def cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        return dot_product / (norm_vector1 * norm_vector2)
    
    # 코사인 유사도 계산
    similar_sentences = []
    for index, row in df.iterrows():
        embedding = row['Embedding']
        similarity = cosine_similarity(target_vector, embedding)
        similar_sentences.append((row['Title'], row['Date'], row['Content'], similarity))

    # 유사도가 높은 순서대로 정렬
    similar_sentences.sort(key=lambda x: x[3], reverse=True)

    # 뽑은 text 전처리 함수
    def processing_text(text):
        content_text = BeautifulSoup(text, "html5lib").get_text() # HTML 태그 제거
        content_text = re.sub("\n", " ", content_text)
        return content_text

    # 1번째와 2번째 본문에 대한 전처리
    similar_sentences[0][2] = processing_text(similar_sentences[0][2])
    similar_sentences[1][2] = processing_text(similar_sentences[1][2])

    return similar_sentences[:2] # 첫번째, 두번째 기사정보 list 형태




