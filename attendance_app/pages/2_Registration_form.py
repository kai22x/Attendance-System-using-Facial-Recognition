import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')
registration_form = face_rec.RegistrationForm()

#STEP 1:COLLECT PERSON NAME AND ROLE
#FORM

person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select Your Role',options=('Student',
                                                      'Teacher'))
#STEP 2: COLLECT FACIAL EMBEDDING OF THAT PERSON
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24")
    reg_img,embedding = registration_form.get_embeddings(img)
    #TWO STEP PROCESS: 1ST STEP TO SAVE DATA INTO LOCAL COMPUTER IN .TXT
    if embedding is not None:
        with open ('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")
webrtc_streamer(key='registration', video_frame_callback=video_callback_func)

#SAVE 3 : SAVE THE DATA IN THE REDIS DATABASE
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f"{person_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')

    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again')

