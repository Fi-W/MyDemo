import streamlit as st
from main import Translation

st.set_page_config(page_title="E-C Translation (Short Sentence)", page_icon="📒")

st.sidebar.header("NLP: E-C Translation")

st.title("NLP - 英汉翻译")
st.header("NLP - E-C Translation Demo")


st.markdown(
    """ 
       
    **Notice**:   
    This is a simple demo of NLP for English to Chinese translation.  
    The model is trained on a dataset of 20,000 English to Chinese short sentences. So only short sentences can be well-processed currently. 
      

"""
)

st.divider()

translation = Translation()

txt = st.text_input(label="##### 请输入英文  Please input an English sentence ", placeholder="请输入英文... Please input an English sentence ...")

btn = st.button("提交 Submit")



if btn:
    
    final_result = translation.infer(sentence=txt)
    out = ''.join(final_result)
    st.info(out)
   