import cv2
import mediapipe as mp
import numpy as np
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
from concurrent.futures import ThreadPoolExecutor
import random
from vosk import SetLogLevel
import time

EYE_LENGTH = 11.7 # mm
F = 100 # 65->100

DPI = 900/1050 * 124
INCH2MM = 25.4
DPMM = DPI/INCH2MM # dots par mm

WINDOW_SIZE = 500

q = queue.Queue()
# SetLogLevel(-1)

def main():
    # 深度推定モデルのインスタンス化
    model = mp.solutions.face_mesh.FaceMesh( max_num_faces=1, refine_landmarks=True)

    # カメラ設定
    cap = cv2.VideoCapture(0)
    # 音声入力デバイス設定
    device_info = sd.query_devices(0, "input")
    samplerate = int(device_info["default_samplerate"])
    model_vosk = Model('model_jp')

    # 視力のデフォルト設定
    score = 1.0
    correct_count = 0

    # 最初は1.0スタート．間違えるごとに視力を下げていき，2回連続で成功したらスコアを表示
    while True:

        truth = random.choice(['up','down','left','right'])

        # テストの区切りタイミングで白の画面を描画
        img = 255*np.ones((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8)
        cv2.imshow('Face detection', img)
        cv2.waitKey(1)

        # テストの実施
        print(f'現在のテスト：視力{score}')
        answer = run_one_test(cap,model,truth,score,samplerate,model_vosk)

        # 間違えた時
        if(answer!=truth):
            # 0.01を2回連続で間違えた時
            if(score==0.01):
                print(f'視力は{score}以下です．')
                return
        
            truth = random.choice(['up','down','left','right'])
            correct_count = 0  
            # 0.1より大きい時は0.1刻みで調節
            if(score>0.1):
                score -= 0.1
                score = round(score,1)
            # 0.1以下は0.01刻みで調節
            else:
                score -= 0.01
                score = round(score,2)
        
        # 正解した時
        else:
            correct_count+=1

            # 2回連続で正解したとき
            if(correct_count==2):
                print(f'あなたの視力は{score}です．')
                return
            
            # 再挑戦
            else:
                truth = random.choice(['up','down','left','right'])
        
        # print(score,correct_count)

def run_one_test(cap,model,truth,score,samplerate,model_vosk):
    # スレッド作成
    executor = ThreadPoolExecutor()
    future = executor.submit(run_vosk,samplerate,model_vosk)

    NUM_WINDOW = 5
    length_list = [0 for k in range(NUM_WINDOW)]

    # 描画ループ
    while True:
        # ビデオフレームの取得
        success, image = cap.read()
        ih = image.shape[0]
        iw = image.shape[1]
        if not success:
            break
        
        # 空画像の準備
        img = 255*np.ones((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8)
        
        # モデルの適用
        results = model.process(image)

        # 結果の表示
        if results.multi_face_landmarks:
            # 画面内に二人以上入ってしまっている時
            if(len(results.multi_face_landmarks)!=1):
                print('テストは一人で行ってください．')
            else:
                # ランドマーク取得
                face_landmarks = results.multi_face_landmarks[0]
                # 虹彩のピクセルとf値より距離を推定
                length_list.append((EYE_LENGTH * F)/calc_iris_pixels(face_landmarks.landmark[468:478],iw,ih))
                length_list = length_list[1:]
                length = np.mean(length_list)
                # length = (EYE_LENGTH * F)/calc_iris_pixels(face_landmarks.landmark[468:478],iw,ih)

                # 音声認識が完了済みのとき
                if future.done():
                    # 結果が上下左右でない時
                    if future.result()=='':
                        print('もう一度入力してください．')
                        future = executor.submit(run_vosk,samplerate,model_vosk)
                    # 結果が正しく取れている時
                    else:
                        # print(future.result())
                        return future.result()
                        print(future.result())

                # ランドルト環のプロット
                plot_landolts(img, score, length, truth)
                # plot_landolts(img, Landolts[1.0],500)

        key = cv2.waitKey(1)
        # ESCで終了
        if(key == 27):  # ESC
            break
        # # 矢印キーも対応
        if(key==0):
            return 'up'
        if(key==1):
            return 'down'
        if(key==3):
            return 'right'
        if(key==2):
            return 'left'

        cv2.imshow('Face detection', img)

def calc_iris_pixels(landmark,iw,ih):
    # 0:左
    # 1:左右
    # 2:左上
    # 3:左左
    # 4:左下
    # 5:右
    # 6:右右
    # 7:右上
    # 8:右左
    # 9:右下
    # 
    l_yoko = np.linalg.norm(np.array([landmark[1].x*iw,landmark[1].y*ih])-np.array([landmark[3].x*iw,landmark[3].y*ih]))
    l_tate = np.linalg.norm(np.array([landmark[2].x*iw,landmark[2].y*ih])-np.array([landmark[4].x*iw,landmark[4].y*ih]))
    r_yoko = np.linalg.norm(np.array([landmark[6].x*iw,landmark[6].y*ih])-np.array([landmark[8].x*iw,landmark[8].y*ih]))
    r_tate = np.linalg.norm(np.array([landmark[7].x*iw,landmark[7].y*ih])-np.array([landmark[9].x*iw,landmark[9].y*ih]))
    return np.mean([l_yoko,l_tate,r_yoko,r_tate])

def plot_landolts(img,score,length,answer):
    # 環の直径(mm) = 0.015 x カメラまでの距離(cm)/視力(0.01~1.0)
    
    mm_o = 0.015* length / score # 直径(mm)

    cv2.circle(img,
           center=(WINDOW_SIZE//2, WINDOW_SIZE//2),
           radius=int(np.round(mm_o/2*DPMM)),
           color=(0, 0, 0),
           thickness=int(np.round(mm_o/5*DPMM)),
           lineType=cv2.LINE_AA)
    if(answer=='up'):
        cv2.rectangle(img, 
                    (int(np.round(WINDOW_SIZE/2-DPMM*mm_o/10)), 0), 
                    (int(np.round(WINDOW_SIZE/2+DPMM*mm_o/10)), int(np.round(WINDOW_SIZE/2))), 
                    (255, 255, 255), thickness=-1)
    if(answer=='down'):
        cv2.rectangle(img, 
                    (int(np.round(WINDOW_SIZE/2-DPMM*mm_o/10)), int(np.round(WINDOW_SIZE/2))), 
                    (int(np.round(WINDOW_SIZE/2+DPMM*mm_o/10)), int(np.round(WINDOW_SIZE))), 
                    (255, 255, 255), thickness=-1)
    if(answer=='right'):
        cv2.rectangle(img, 
                    (int(np.round(WINDOW_SIZE/2)),int(np.round(WINDOW_SIZE/2-DPMM*mm_o/10))), 
                    (int(np.round(WINDOW_SIZE)),int(np.round(WINDOW_SIZE/2+DPMM*mm_o/10))), 
                    (255, 255, 255), thickness=-1)
    if(answer=='left'):
        cv2.rectangle(img, 
                    (0,int(np.round(WINDOW_SIZE/2-DPMM*mm_o/10))), 
                    (int(np.round(WINDOW_SIZE/2)),int(np.round(WINDOW_SIZE/2+DPMM*mm_o/10))), 
                    (255, 255, 255), thickness=-1)
    cv2.putText(img,text=f'length:{np.round(length)}cm, score:{score}',org=(50, 50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0),thickness=1,lineType=cv2.LINE_AA)

def format_text(txt):
    if txt=='し た':
        return 'down'
    elif '右' in txt:
        return 'right'
    elif '左' in txt:
        return 'left'
    elif '上' in txt:
        return 'up'
    else:
        return ''

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def run_vosk(samplerate, model):

    with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=0,
            dtype="int16", channels=1, callback=callback):

        rec = KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                return format_text(json.loads(rec.Result())['text'])
                
                # for vosk_test()
                # return json.loads(rec.Result())['text']

if __name__=='__main__':
    main()