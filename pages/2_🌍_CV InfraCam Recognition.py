import streamlit as st
from streamlit_option_menu import option_menu 
import os
import numpy as np
import cv_utils as utils
from ultralytics import YOLO    # type: ignore
import tempfile
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st


st.set_page_config(page_title="CV InfraCam Detection", page_icon="🌍")
st.sidebar.header("CV: Infrared Camera Detection of Human and Vehicle")

model = YOLO('./InfraCamR/best.pt')

# @st.cache(show_spinner=False)
def load_local_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()  
    image = np.array(Image.open(BytesIO(bytes_data)))
    return image

 

#定义边栏导航
with st.sidebar:
    choose = option_menu('请选择 Please choose',['视频处理 Video','图片处理 Image'],
                         icons=['camera-video-fill','image'])
    
if choose == '视频处理 Video':
        st.title('红外摄像头人车检测项目')
        st.header("Infrared Camera Detection of Human and Vehicle")
#        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown(
        """ 
        
        **Notice**:   
        This is a demo of computer vision model for infrared camera detection of human and vehicle.    
        The model is trained on a dataset of 30788 images and 29970 videos. 
        YOLO (v8n) is used in the training and testing process and a best model is obtained.  
        The current accuracy is around 0.69.  
        

        """
        )
        st.divider()

        tab1, tab2 = st.tabs(['案例效果 Example', '视频处理 Video'])
        with tab1:
            # 创建两个并排的列
            col1, col2 = st.columns(2)

            # 在第一列中播放原始视频
            with col1:
                st.header("原始视频 Original Video")
                st.video('./InfraCamR/static/traffic_night_HD.mp4')

            # 在第二列中播放处理后的视频
            with col2:
                st.header("处理后的视频 Processed Video")
                st.video('./InfraCamR/static/traffic_night.mp4')

        result_video_dir = None
        with tab2:
            # 创建两个并排的列
            col1, col2 = st.columns(2)

            # 在第一列中上传原始视频
            uploaded_video_file = None
            with col1:
                st.header("原始视频 Original Video")
                # 创建上传视频文件的组件
                uploaded_video_file = st.file_uploader("上传 Upload", type=['mp4', 'avi'])

                if uploaded_video_file is not None:
                   
                    # 展示上传的视频文件
                    st.video(uploaded_video_file)
                    

            # 在第二列中展示处理后的视频
            with col2:
                st.header("处理后的视频 Processed Video")
                if uploaded_video_file is not None:
                  

                    # st.video(result_video_dir)
                    with st.spinner("Running..."):
                        try:
                            tfile = tempfile.NamedTemporaryFile()
                            tfile.write(uploaded_video_file.read())
                            vid_cap = cv2.VideoCapture(tfile.name)

                            st_frame = st.empty()

                            while (vid_cap.isOpened()):
                                success, image = vid_cap.read()
                                if success:
                                    utils.display_detected_frames(model, st_frame, image)
                                else:
                                    vid_cap.release()
                                    break
                        except Exception as e:
                            st.error(f"Error loading video: {e}")


elif choose == '图片处理 Image':
        st.title('红外摄像头人车识别项目')
        st.header("Infrared Camera Recognition of Human and Vehicle")
#        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
        """ 
        
        **Notice**:   
        This is a demo of computer vision model for infrared camera detection of human and vehicle.    
        The model is trained on a dataset of 30788 images and 29970 videos. 
        YOLO (v8n) is used in the training and testing process and a best model is obtained.  
        The current accuracy is around 0.69.  
        

        """
        )

        st.divider()

        tab1, tab2 = st.tabs(['案例效果 Example', '图片处理 Image'])

        with tab1:
            # 创建两个并排的列
            col1, col2 = st.columns(2)

            # 在第一列中展示原始图像
            with col1:
                st.header("原始图片 Original Image")
                st.image('./InfraCamR/static/rgb_1002.jpg')

            # 在第二列中播放处理后的图片
            with col2:
                st.header("处理后的图片 Processed Image")
                st.image('./InfraCamR/static/rgb_1002_detect.jpg')
        
        # 处理后的图片
        result_img_dir = None
        with tab2:
            # 创建两个并排的列
            col1, col2 = st.columns(2)

            # 在第一列中上传原始图片
            with col1:
                st.header("原始图片 Original Image")
                # 创建上传图片文件的组件
                uploaded_file = st.file_uploader("上传 Upload", type=['jpg', 'png'])
                

                if uploaded_file is not None:

                    # 展示上传的图片文件
                    st.image(uploaded_file)
                
            # 在第二列中展示处理后的图片
            with col2:
                st.header("处理后的图片 Processed Image")
                if uploaded_file is not None:
                    image = load_local_image(uploaded_file)
                    result_img = utils.predict_img(model, image)
                    
                    # 展示处理后的图片
                    st.image(result_img)    
