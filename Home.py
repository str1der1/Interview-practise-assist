import streamlit as st
import pandas as pd
import numpy as np

# import src.CvUpdateReview as cv
import src.InterviewPrep_new_with_langgraph as ip
# import src.gmailSummary as gm


st.set_page_config(
    page_title="Aftab Home Streamlit Page",
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
              This is the introduction page. USe the dropdown to navigate to different demo Apps
              """)
    

def plotting_demo():
    st.title("Plotting Demo")
    st.write("HEre, we crreate a simple demo")

    chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    st.line_chart(chart_data)

def mapping_demo():
    st.title("Mapping Demo")
    st.write("This demo is to show a map with random points")

    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=["lat", "lon"]
    )

    st.map(map_data)

#Dictionary to map page names to functions 
page_names_to_funcs = {
    "-": intro,
    "Example - Plotting Demo": plotting_demo,
#    "Example - Mapping demo" : mapping_demo,
#    "CV Analysis": cv.cv_analysis,
    "Interview Question Prep (with langgraph)": ip.interview_question_evaluator2_with_langgraph
#    "Gmail reviewer": gm.gmail_viewer
}

# Create sidebar to show thevarious options 
selected_page = st.sidebar.selectbox("Choose a page", options=page_names_to_funcs.keys())

# Run the function to open that page
page_names_to_funcs[selected_page]()
