import streamlit as st
import pandas as pd
import joblib
import jieba

st.set_page_config(page_title="ML User Comment Analysis", page_icon="📈")

st.sidebar.header("Machine Learning: Hotel Review Analysis")

st.title("机器学习 - 酒店评论情感分析展示")
st.header("Machine Learning - Hotel Review Analysis Demo")


st.markdown(
    """ 
       
    **Notice**:   
    This is a simple demo of machine learning model for sentiment analysis, mainly focuses on hotel user comments.  
    The current accuracy is around 0.89 while performs a bit better for longer texts than shorter ones.  
    There is only two classification for now, positive and negative without neutral. So for comments contain both positive and negative parts, the accuracy might be lower.  
      

"""
)

st.divider()

txt = st.text_input(label="##### 请输入评论  Please input your comment ", placeholder="请输入评论... Please input your comment...")

btn = st.button("提交 Submit")


if btn:
    
    # 调用模型
    clf = joblib.load("./MLcommentA/clf.pkl")
    vect = joblib.load("./MLcommentA/vect.pkl")
    
    # 数据处理
    comment = [' '.join(jieba.cut(txt))]
    X_try = vect.transform(comment)
    
    # 预测结果
    y_pred = clf.predict(X_try.toarray())
    
    # 输出结果
    if y_pred == 0:
        st.info("感谢您的反馈！ Thank you very much for your feedback.")
    elif y_pred == 1:
        st.info("感谢您的好评！ Thank you very much for your positive feedback!")
    

