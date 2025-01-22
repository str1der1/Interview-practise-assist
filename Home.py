import streamlit as st
import pandas as pd
import numpy as np

import src.InterviewPrep as ip


st.set_page_config(
    page_title="Home Streamlit Page  (see sidebar for Apps)",
    # page_icon="🧊",
    layout="wide",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)



def intro():
    st.title("Welcome to my Home Page for my App")
    st.write ("""
              This is the introduction page. USe sidebar dropdown to navigate to different Apps
              """)
    

def plotting_demo():
    st.title("Plotting Demo")
    st.write("HEre, we crreate a simple demo")

    chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    st.line_chart(chart_data)


#Dictionary to map page names to functions 
page_names_to_funcs = {
    "-": intro,
    "Example - Plotting Demo": plotting_demo,
    "Interview Question Prep": ip.interview_question_prep
}

# Create sidebar to show thevarious options 
selected_page = st.sidebar.selectbox("Choose a page", options=page_names_to_funcs.keys())

# Run the function to open that page
page_names_to_funcs[selected_page]()
