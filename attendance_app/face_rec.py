import numpy as np
import pandas as pd
import cv2
import redis
import streamlit as st
# Insight Face
from insightface.app import FaceAnalysis  # Corrected import statement
from sklearn.metrics import pairwise

#time
import time 
from datetime import datetime
import os

#connect to Redis Client

hostname ='redis-13192.c321.us-east-1-2.ec2.cloud.redislabs.com'
port =13192
password ='0D02kYQ6MnMOuI8JdwZfITKsI43TnU6M'

r=redis.StrictRedis(host=hostname,port=port, password=password)

#Retrive Data from database

#name = 'academy:register'
def retrive_data(name):
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrive_df= retrieve_series.to_frame().reset_index()
    retrive_df.columns = ['name_role','facial_features']
    retrive_df[['Name','Role']] = retrive_df['name_role'].apply(lambda x : x.split('@')).apply(pd.Series)
    return retrive_df[['Name', 'Role', 'facial_features']]

#configure face analysis

faceapp = FaceAnalysis(name='buffalo_sc',
                     root='insightface_model',
                     providers=['CPUExecutionProvider'])

faceapp.prepare(ctx_id=0,det_size=(640, 640), det_thresh = 0.5)

# ML SEARCrch_algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name', 'Role'], thresh=0.5):
    """Cosine simility base search Algoritm"""
    #step 1 : take the data frame(collection of data)
    dataframe = dataframe.copy()
    #step 2 : Index face embedding from the database and convert into array
    x_list = dataframe[feature_column].tolist()
    x = np.asarray(x_list)

    #step 3 : calculate cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    #step 4 : Filter the data and get the person name
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()  # Fix the variable name from datafilter to data_filter
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
    return person_name, person_role

#Real Time Prdiction
#we need to save logs for every 1 min
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def saveLogs_redis(self):
        #step1:  create a logs datafram
        dataframe = pd.DataFrame(self.logs)

        #step2: drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name',inplace=True)

        #step3: push data to redis database (list)
        #encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data =[]
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush('attendance_logs',*encoded_data)

        self.reset_dict()

 
    def face_prediction(self,img_test,dataframe, feature_column,name_role=['Name', 'Role'], thresh=0.5):
        #step 0: find the time

        current_time = str(datetime.now())

        #step 1: tske the test image and apply to insight face
        results = faceapp.get(img_test)
        img_test =img_test.copy()
        #step 2 : use for loop and extract each embedding and pass to ml_search _algorithm
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embed_test = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe, feature_column, test_vector=embed_test, name_role=name_role, thresh=thresh)
                
            if person_name == 'Unknown':
                colour = (0, 0, 255)
            else:
                colour = (0, 255, 0)
                    
            cv2.rectangle(img_test, (x1, y1), (x2, y2), colour)
            text_gen = person_name
            cv2.putText(img_test, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, 1)
            cv2.putText(img_test,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX, 0.5, colour, 1)
            #save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
        return img_test
#REGISTRATION FORM
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
    def get_embeddings(self,frame):
        #GET RESULTS FROM INSIGHTFACE MODEL
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1

            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
            text = f"sample= {self.sample}"
            cv2.putText(frame,text,(x1, y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)


            #FACIAL FEATURES
            embeddings = res['embedding']
        return frame,embeddings
    def save_data_in_redis_db(self,name,role):
        #validation name
        if name is not None:
            if name.strip()!='':
                key= f'{name}@{role}'
            else:
                return 'name_false'
        else:
                return 'name_false'

        #if face_embedding.txt exits
        if 'face_embedding.txt' not in os.listdir():
                    return  'file_false'
        

        #step1: load "face_embedding.txt"
        x_array=np.loadtxt('face_embedding.txt',dtype=np.float32) #flatten array

        #step2: convert into array
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)

        #step3: call mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        #step4: save this into redis database
        #redis hashes
        r.hset(name='academy:register',key=key,value=x_mean_bytes)


        #
        os.remove('face_embedding.txt')
        self.reset()

        return True




        



