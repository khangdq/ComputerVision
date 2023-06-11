import streamlit as st
# import io
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import requests
# from io import BytesIO
# import pandas as pd
# from tensorflow.keras.preprocessing import image
import cv2
from ultralytics import YOLO
# import os
# import shutil
# @st.cache_resource
# def load_modelClass():
#     classifier=tf.keras.models.load_model('./Model/Project1_PreTrain_mobilenet_smallsize.h5')
#     return classifier
# @st.cache_resource
# def load_modelSeg():
#     folderPath = 'runs/segment'    
#     # Check Folder is exists or Not
#     if os.path.exists(folderPath):     
#         # Delete Folder code
#         shutil.rmtree(folderPath)
#     model = YOLO("./Model/bestseg.pt")
#     return model
# @st.cache_resource
# def load_modelDet():
#     folderPath = 'runs/detect'    
#     # Check Folder is exists or Not
#     if os.path.exists(folderPath):     
#         # Delete Folder code
#         shutil.rmtree(folderPath)
#     model = YOLO("./Model/bestdet.pt")
#     return model
# def convert_df(df):
#    return df.to_csv(index=False).encode('utf-8')
# def predictfunc(classifier,test_image):
#     test_image = image.img_to_array(test_image)
#     #print(test_image.shape)
#     test_image = test_image/255
#     test_image = np.expand_dims(test_image, axis = 0) 
#     #print(test_image.shape)

#     result = classifier.predict(test_image,verbose = 0)
#     #print(result)
#     prediction=""
#     if result[0][0]==np.max(result[0]): 
#         prediction = ":green[Glass]"
#     elif result[0][1]==np.max(result[0]):
#         prediction = ":blue[Metal]"
#     elif result[0][2]==np.max(result[0]):
#         prediction = ":orange[Paper]"
#     else:
#         prediction = ":violet[Plastic]"
#     return prediction
# def display_images(listimage,classifier,max=4):
#     similar = len(listimage)
#     df = pd.DataFrame(data={'No':[],'Name':[],'Predict':[]})
#     with st.container():
#         k=0
#         for i in range(int(round(similar/max+0.5,0))):
#             columns = st.columns(max)
#             for column in range(max):
#                 k=k+1
#                 image1 = listimage[column+max*i]
#                 bytes_data = image1.read()
#                 image2 = Image.open(io.BytesIO(bytes_data))
#                 cell = columns[column] 
#                 cell.image(image2.resize((300,300)))
#                 cell.write(image1.name)
#                 result='Can not predict this image'
#                 try:
#                     result=predictfunc(classifier,image2.resize((224,224)))
#                     cell.markdown(result)
#                     result=result.split("[")[1].split("]")[0]
#                     #cell.success(result, icon="✅")
#                 except:
#                     cell.error('Error!', icon="🚨")
#                 df = df.append({'No':k,'Name':image1.name,'Predict':result}, ignore_index=True)
#                 cell.write(" ")
#                 cell.write(" ")
#                 if k==similar:
#                     break
#             if k==similar:
#                 break
#     return df
# def display_images_seg(listimage,model,max=4,conf=0.2):
#     similar = len(listimage)
#     with st.container():
#         k=0
#         for i in range(int(round(similar/max+0.5,0))):
#             columns = st.columns(max)
#             for column in range(max):
#                 k=k+1
#                 image1 = listimage[column+max*i]
#                 image2 = Image.open(image1)
#                 cell = columns[column] 
#                 try:
#                     res = model.predict(image2.resize((224,224)), conf=conf, line_width=2)
#                     #boxes = res[0].boxes
#                     res_plotted = res[0].plot()[:, :, ::-1]
#                     cell.image(res_plotted)  
#                     cell.write(image1.name)
#                 except:
#                     cell.write(image1.name)
#                     cell.error('Error!', icon="🚨")
#                     cell.write(" ")
#                     cell.write(" ")
#                 if k==similar:
#                     break
#             if k==similar:
#                 break
# def display_images_link(listimage,classifier,max=4):
#     similar = len(listimage)
#     df = pd.DataFrame(data={'No':[],'Link':[],'Predict':[]})
#     with st.container():
#         k=0
#         for i in range(int(round(similar/max+0.5,0))):
#             columns = st.columns(max)
#             for column in range(max):
#                 k=k+1
#                 link = listimage[column+max*i]
#                 ok=0
#                 cell = columns[column] 
#                 try:
#                     response = requests.get(link)
#                     image2 = Image.open(BytesIO(response.content))         
#                     cell.image(image2.resize((300,300)))
#                     ok=1   
#                 except:
#                     cell.image(Image.open('Images/canload.jpg').resize((300,300)))
#                     cell.error('Link error!', icon="🚨")
#                     result='Link error'
#                 if ok==1:    
#                     try:
#                         result='Can not predict from this link'
#                         result=predictfunc(classifier,image2.resize((224,224)))
#                         cell.markdown(result)
#                         result=result.split("[")[1].split("]")[0]
#                         #cell.success(result, icon="✅")
#                     except:
#                         cell.error('Error!', icon="🚨")
#                 df = df.append({'No':k,'Link':link,'Predict':result}, ignore_index=True)
#                 cell.write(" ")
#                 cell.write(" ")
#                 if k==similar:
#                     break
#             if k==similar:
#                 break
#     return df
# def display_images_link_seg(listimage,model,max=4,conf=0.2):
#     similar = len(listimage)
#     with st.container():
#         k=0
#         for i in range(int(round(similar/max+0.5,0))):
#             columns = st.columns(max)
#             for column in range(max):
#                 k=k+1
#                 link = listimage[column+max*i]
#                 ok=0
#                 cell = columns[column] 
#                 try:
#                     response = requests.get(link)
#                     image2 = Image.open(BytesIO(response.content))         
#                     ok=1   
#                 except:
#                     cell.image(Image.open('Images/canload.jpg').resize((300,300)))
#                     cell.error('Link error!', icon="🚨")
#                 if ok==1:    
#                     try:
#                         res = model.predict(image2.resize((224,224)), conf=conf, line_width=2)
#                         #boxes = res[0].boxes
#                         res_plotted = res[0].plot()[:, :, ::-1]
#                         cell.image(res_plotted) 
#                     except:
#                         cell.image(image2.resize((300,300)))
#                         cell.error('Error!', icon="🚨")
#                 cell.write(" ")
#                 cell.write(" ")
#                 if k==similar:
#                     break
#             if k==similar:
#                 break


# ### Load model
# # GUI

# st.title("Machine Learning Project")
# st.write("# Computer Vision")
# st.write("## Object Classification")
# st.write("## Object Detection- Object Segmentation")
# menu = ["1. Introduction","2. Model Classification", "3. Model Object Segmentation","4. Prediction"]
# choice = st.sidebar.selectbox('Menu', menu)
# if choice == '1. Introduction':
#     st.write("## 4. Prediction")
#     genre = st.radio(
#     "Select the type of input data",
#     ('Image', 'Video'))
#     with st.expander("Setting"):
#         conf = st.slider('Object confidence threshold for detection', 0.1, 1.0, 0.4)
#         maxcol = st.slider('Number of columns to display', 3, 6, 4)
#     if genre == 'Image':
#         st.info('Image')
#         genre1 = st.radio(
#         "Select Select upload type",
#         ('Upload image', 'Upload link','Upload file','Camera'))
#         if genre1=='Upload image':
#             uploaded_files = st.file_uploader("Upload images",type=['jpg','jpeg','png'],help="Upload images in jpg, jpeg, png format", accept_multiple_files=True)
#         if genre1=='Upload link':
#             link = st.text_input('Input link', '')
#         if genre1=='Upload file':
#             df2 = st.file_uploader("Upload file ", type={"csv"})
#         if genre1=='Camera':
#             picture = st.camera_input("Take a picture")
#         genre2 = st.radio(
#         "Select model",
#         ("Object Classification", "Object Detection","Object Segmentation"))
#         if genre2=="Object Classification": #Object Classification
#             classifier=load_modelClass()
#             if genre1=='Upload image':
#                 if np.size(uploaded_files)>0: 
#                     ## Download kết quả\                              
#                     st.info('Object Classification results:')
#                     df=convert_df(display_images(uploaded_files,classifier,maxcol)) 
#                     st.download_button("Download result",df,"file.csv","text/csv",key='download-csv')  
#                 else:
#                     st.warning('Please upload images!', icon="⚠️")
#             if genre1=='Upload link':
#                 if link!="":
#                     try:
#                         response = requests.get(link)
#                         image1 = Image.open(BytesIO(response.content))
#                         st.info('Object Classification results:')
#                         st.image(image1.resize((300,300)))
#                         try:
#                             result=predictfunc(classifier,image1.resize((224,224)))
#                             st.markdown(result)
#                             result=result.split("[")[1].split("]")[0]
#                             ## Download kết quả\    
#                             df=pd.DataFrame(data={'No':[1],'Link':[link],'Predict':[result]})                          
#                             df=convert_df(df) 
#                             st.download_button("Download result ",df,"file.csv","text/csv",key='download-csv') 
#                         except:
#                             st.error('Error!', icon="🚨")
                        
#                     except:
#                         st.error('Can not load image from this link!', icon="🚨")
#                 else:
#                     st.warning('Link can not null!', icon="⚠️")
#             if genre1=="Upload file":
#                 if df2 is not None:
#                     try:
#                         df2 = pd.read_csv(df2)
#                         st.dataframe(df2)
#                         ok=0
#                         try: 
#                             data=df2.Link
#                             ok=1
#                         except:
#                             st.error('Link column not found in file!', icon="🚨") 
#                         if ok==1:
#                             df=convert_df(display_images_link(df2.Link.values,classifier,maxcol))
#                             st.download_button("Download result",df,"file.csv","text/csv",key='download-csv') 
#                     except:
#                         st.error('Can not load this file!', icon="🚨") 
#             if genre1=='Camera':
#                 if picture is not None:
#                     try:
#                         image1 = Image.open(BytesIO(picture.getvalue()))
#                         st.info('Object Classification results:')
#                         st.image(image1)
#                         result=predictfunc(classifier,image1.resize((224,224)))
#                         st.markdown(result)
#                     except:
#                         st.error('Error!', icon="🚨")
#         if genre2=="Object Detection": #Object Detection
#             model=load_modelDet()
#             if genre1=='Upload image':
#                 if np.size(uploaded_files)>0:                              
#                     st.info('Object Detection results:')
#                     display_images_seg(uploaded_files,model,maxcol,conf)
#                 else:
#                     st.warning('Please upload images!', icon="⚠️")
#             if genre1=='Upload link':
#                 if link!="":
#                     try:
#                         response = requests.get(link)
#                         image1 = Image.open(BytesIO(response.content))
#                         st.info('Object Detection results:')
#                         try:
#                             res = model.predict(image1.resize((300,300)), conf=conf, line_width=2)
#                             #boxes = res[0].boxes
#                             res_plotted = res[0].plot()[:, :, ::-1]
#                             st.image(res_plotted) 
#                         except:
#                             st.error('Error!', icon="🚨")
                        
#                     except:
#                         st.error('Can not load image from this link!', icon="🚨")
#                 else:
#                     st.warning('Link can not null!', icon="⚠️")   
#             if genre1=="Upload file":
#                 if df2 is not None:
#                     try:
#                         df2 = pd.read_csv(df2)
#                         st.dataframe(df2)
#                         ok=0
#                         try: 
#                             data=df2.Link
#                             ok=1
#                         except:
#                             st.error('Link column not found in file!', icon="🚨") 
#                         if ok==1:
#                             display_images_link_seg(df2.Link.values,model,maxcol,conf)
                            
#                     except:
#                         st.error('Can not load this file!', icon="🚨")
#             if genre1=='Camera':
#                 if picture is not None:
#                     try:
#                         image1 = Image.open(BytesIO(picture.getvalue()))
#                         st.info('Object Detection results:')
#                         res = model.predict(image1, conf=conf, line_width=5)
#                         #boxes = res[0].boxes
#                         res_plotted = res[0].plot()[:, :, ::-1]
#                         st.image(res_plotted) 
#                     except:
#                         st.error('Error!', icon="🚨")
#         if genre2=="Object Segmentation": #Object Segmentation
#             model=load_modelSeg()
#             if genre1=='Upload image':
#                 if np.size(uploaded_files)>0:                              
#                     st.info('Object Detection results:')
#                     display_images_seg(uploaded_files,model,maxcol,conf)
#                 else:
#                     st.warning('Please upload images!', icon="⚠️")
#             if genre1=='Upload link':
#                 if link!="":
#                     try:
#                         response = requests.get(link)
#                         image1 = Image.open(BytesIO(response.content))
#                         st.info('Object Detection results:')
#                         try:
#                             res = model.predict(image1.resize((300,300)), conf=conf, line_width=2)
#                             #boxes = res[0].boxes
#                             res_plotted = res[0].plot()[:, :, ::-1]
#                             st.image(res_plotted) 
#                         except:
#                             st.error('Error!', icon="🚨")
                        
#                     except:
#                         st.error('Can not load image from this link!', icon="🚨")
#                 else:
#                     st.warning('Link can not null!', icon="⚠️")   
#             if genre1=="Upload file":
#                 if df2 is not None:
#                     try:
#                         df2 = pd.read_csv(df2)
#                         st.dataframe(df2)
#                         ok=0
#                         try: 
#                             data=df2.Link
#                             ok=1
#                         except:
#                             st.error('Link column not found in file!', icon="🚨") 
#                         if ok==1:
#                             display_images_link_seg(df2.Link.values,model,maxcol,conf)
                            
#                     except:
#                         st.error('Can not load this file!', icon="🚨")
#             if genre1=='Camera':
#                 if picture is not None:
#                     try:
#                         image1 = Image.open(BytesIO(picture.getvalue()))
#                         st.info('Object Detection results:')
#                         res = model.predict(image1, conf=conf, line_width=5)
#                         #boxes = res[0].boxes
#                         res_plotted = res[0].plot()[:, :, ::-1]
#                         st.image(res_plotted) 
#                     except:
#                         st.error('Error!', icon="🚨")            
#     else:
#         st.info("Video")
#         uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
#         temporary_location = False
        
#         if uploaded_file is not None:
#             g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
#             temporary_location = "video.mp4"

#             with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
#                 out.write(g.read())  ## Read bytes into file

#             # close file
#             out.close()
#             video_file = open('video.mp4', 'rb')
#             video_bytes = video_file.read()
#             st.video(video_bytes)
#             video_file.close()
#             folderPath = 'Videos'    
#             genre3 = st.radio(
#             "Select model",
#             ( "Object Detection","Object Segmentation"))
#             try:
#                 if genre3=="Object Detection":
#                     model=load_modelDet()
#                     save_path = 'runs/detect/predict/video.mp4'
#                     # Check Folder is exists or Not
#                     if os.path.exists(folderPath):     
#                         # Delete Folder code
#                         shutil.rmtree(folderPath)
#                     # Check whether the specified path exists or not
#                     isExist = os.path.exists(folderPath)
#                     if not isExist:
#                         # Create a new directory because it does not exist
#                         os.makedirs(folderPath)
#                 elif genre3=="Object Segmentation":
#                     model=load_modelSeg()
#                     save_path = 'runs/segment/predict/video.mp4'
#                     # Check Folder is exists or Not
#                     if os.path.exists(folderPath):     
#                         # Delete Folder code
#                         shutil.rmtree(folderPath)
#                     # Check whether the specified path exists or not
#                     isExist = os.path.exists(folderPath)
#                     if not isExist:
#                         # Create a new directory because it does not exist
#                         os.makedirs(folderPath)
#                 with st.spinner('predicting...'):
#                     model.predict(source='video.mp4', imgsz=1280, save=True, line_width=1,conf=conf)
#                     # Input video path
#                     # Compressed video path
#                     compressed_path = "Videos/compressed_video.mp4"
#                     os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")
#                     st.success('Done!')
#                 video_file = open('Videos/compressed_video.mp4', 'rb')
#                 video_bytes = video_file.read()
#                 video_file.close()
#                 st.video(video_bytes)
#             except:
#                 st.error('Something went wrong! Please refresh the page and try again!', icon="🚨")
