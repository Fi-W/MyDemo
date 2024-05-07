import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)



st.write("# Welcome to My Demo! 👋")

st.sidebar.header("Introduction")
st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This platform is designed to prove my learning ability and willingness in the field of computer science.  

    Along with my progress in this area, I will share more demos and further improve the page design as well as the models' performance.

    
"""
)

st.divider()

st.markdown(
    """
    **👈 Select a demo from the sidebar** to see some examples!  

    ### What I have managed so far?
    - Machine Learning: User comment sentiment analysis (mainly focuses on hotel reviews)
    - Computer Vision: Infrared Camera Detection of Human and Vehicle
    
    ###### More to come!
"""
)



