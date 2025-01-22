
import streamlit as st
import sqlite3
import random

from langchain_openai import ChatOpenAI
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

# from dotenv import load_dotenv
import os

# Load environment variables
# load_dotenv()
# Access your secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
db_url = st.secrets["database_url"]


# Initialise the OpenAI API for audio transcription using Whisper
openAI_client = OpenAI()

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.5,
    openai_api_key=openai_api_key
)

# Connect to SQLite Database
def connect_db():
    # conn = sqlite3.connect("amazonQBank_values_and_questions.db")  # Replace with your DB file
    conn = sqlite3.connect(db_url)  # Replace with your DB file
    return conn

# Fetch competencies
def fetch_competencies():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, theme_name, theme_description FROM themes_tbl")
    data = cursor.fetchall()
    conn.close()
    return data

# Fetch random questions for selected competencies
def fetch_questions(competencies, n=3):
    conn = connect_db()
    cursor = conn.cursor()
    questions = []
    for comp_id in competencies:
        cursor.execute(
            "SELECT question FROM questions_tbl WHERE theme_name = ? ORDER BY RANDOM() LIMIT ?",
            (comp_id, n)
        )
        questions.extend(cursor.fetchall())
    conn.close()
    return [q[0] for q in questions]

# Evaluate answer with LLM
def evaluate_answer(answer, competency, strengths, concerns, question):
    prompt = f"""
    For the provided question, evaluate the following answer provided using STAR approach for answering.  Also consider the provided strengths and concerns.
    Question: {question}
    Answer: {answer}
    Strengths: {', '.join(strengths)}
    Concerns: {', '.join(concerns)}
    
    Provide:
    - The first line of the feedback should include a STAR principle score out of 100.
    - If the answer is too short then inform that the answer is too short.
    - If the the answer is not meeting the STAR principle then inform that the answer is not meeting the STAR principle.
    - If there is no feedback to be given, then do not provide any feedback.
    - Otherwise provide a summary of the overall answer with feedback limited to 1 paragraph.
    - If providing feedback, also provide  improvement options with suggestions to improve the answer. Give at least 1 suggestion for each improvment.
    - Make the feeback concise and to the point.
    """
    prompt2 = f"""
    You are an expert evaluator for interview answers, skilled in assessing responses against the STAR principle (Situation, Task, Action, Result). Your goal is to evaluate the provided answer based on the following criteria, while considering the provided strengths and concerns.

    Instructions:
    Start with a STAR principle score out of 100 in the first line of your response.
    Analyze whether the answer fully addresses the STAR principle:
    If the answer is too short to provide meaningful insights, explicitly state: "The answer is too short."
    If the answer does not meet the STAR principle, explain state "It does not meet STAR" and briefly explain why it fails.
    If the answer is well-structured and complete, acknowledge this.
    Incorporate the provided strengths and concerns in your evaluation. Mention how the answer aligns with the strengths or exacerbates the concerns, if applicable.
    Summarize your feedback in one concise paragraph, highlighting both strengths and weaknesses of the answer.
    If feedback is necessary, provide actionable improvement suggestions:
    List at least one suggestion for each identified weakness.
    Ensure suggestions are clear, practical, and directly linked to the STAR principle or the specific question.
    If no feedback is required, explicitly state: "No feedback needed."
    Inputs:

    Competency: {competency}
    Question: {question}
    Strengths: {strengths}
    Concerns: {concerns}
    Answer: {answer}

    Output Example:
    STAR Score: 85/100.
    Summary Feedback: "The answer covers the STAR principle well, particularly the Situation and Task. However, the Result is vague, and the actions could be described in more detail. Strengths such as clear customer focus are demonstrated, but concerns about leadership clarity remain."
    Suggestions:
    - Clearly describe the results achieved, including specific metrics or outcomes.
    - Provide more detailed actions to highlight your leadership and initiative.
    """

    messages = [
        SystemMessage(content="You are a helpful assistant for interview preparation."),
        HumanMessage(content=prompt2)
    ]
    response = llm(messages)
    return response.content


# Function to transcribe audio using OpenAI's Whisper API
def transcribe_audio(audio_bytes):
    try:
            transcript = openAI_client.audio.transcriptions.create(
                model="whisper-1",
                file = audio_bytes
            )
            return transcript.text    
    except Exception as e:
        return f"Error transcribing audio: {e}"
    


# #######  MAIN function ######

def interview_question_prep():

    # App Layout
    st.title("Interview Preparation Assistant")
    st.write("Prepare for interviews using leadership principles and competencies.")

    # Step 1: Select Competencies  - Give the user the option to select competencies
    st.header("1. Select Leadership Principles or Competencies")
    competencies = fetch_competencies()
    # print(f"fetch competencies are : {competencies} ")
    st.session_state.selected_competencies = st.multiselect(
        "Select competencies:", options=[(c[0], c[1], c[2] ) for c in competencies], format_func=lambda x: x[1]
    )

    # print("competencies selected")
    # print(selected_competencies)
    #print(selected_competencies[0][0])

    # Step 2: Select Questions   - From the themes selected, isolate the competencies and give option to select # of questions
    if st.session_state.selected_competencies:
        # Pull competency and Description from DB pull 
        st.session_state.competencyList = [ (comp[1], comp[2]) for comp in st.session_state.selected_competencies]
        # print(f"CompetencyList:  {competencyList}" )
        for index, comp in enumerate(st.session_state.competencyList):
            st.text_area(f"Description for -- {comp[0]}:", value=f"{comp[1]}" , height=100)

        st.header("2. Select Num of Questions")
        st.session_state.question_count  = st.slider("Number of Questions per competency", min_value=0, max_value=5, value=0)
        st.session_state.pull_questions = st.button("Pull Questions")

    # Step 3 - Pull questions from the database  - given competencies and number of questions,  pull X questions from the DB
    if "pull_questions" in st.session_state and st.session_state.question_count > 0 and st.session_state.pull_questions==True:
        # Isolate just the competency from the list selected 
        competency_ids = [comp[1] for comp in st.session_state.selected_competencies]
        # print(f"Competency_IDs: {competency_ids}")
        # Fetch questions (count per competency) from DB
        st.session_state.questions = fetch_questions(competency_ids, st.session_state.question_count)

    # Based omn Questions varoiable (pulled from DB) for each question,  present a Answer box 
    if "questions" in st.session_state:
        st.header("3. Questions")
        st.session_state.answers = {}
        # for i, question in enumerate(selected_questions):
        for i, question in enumerate(st.session_state.questions):
            st.markdown(f"###### Q{i + 1}):\n{question}")
            # Check if answer was presented already before as history
            if question in st.session_state.answers:
                st.session_state.answers[question] = st.text_area(f"Your answer for Question {i + 1}:", value=st.session_state.answers[question], height=100)
            else:
                st.session_state.answers[question] = st.text_area(f"Your answer for Question {i + 1}:", height=100)
                
                # Add option for transcribing audio instead of writing the answer 
                key_id = f"audio_{i}"
                audio_data = st.audio_input("Record a voice message",key=key_id)

                # if audio_data is not None:
                st.audio(audio_data, format="audio/wav")

                # Transcribe audio when button is clicked
                if st.button("Transcribe Audio",key=f"transcribe_button_{i}"):
                    st.write("Transcribing audio...")
                   
                    transcription = transcribe_audio(audio_data)
                    st.write(transcription)
                    st.session_state.answers[question] = transcription

    # print(f"button state for run_evaluation: {st.session_state.run_evaluation}")

        #  Step 4: Evaluate  Results   -  Before evaluating,  pull the list of strengths and concerns for each question competency
        # if "run_evaluation" in st.session_state:
        if st.button("Run Evaluation on Answers"):
            # print(f"Answers: {st.session_state.answers}")

            st.header("4. Results")
            counter=1
            conn = connect_db()
            cursor = conn.cursor()
            for question, answer in st.session_state.answers.items():   

            #     # Fetch strengths and concerns for the competency
            #     cursor.execute("""
            #         SELECT strengths_tbl.strength, concerns_tbl.concern
            #         FROM questions_tbl
            #         JOIN themes_tbl ON themes_tbl.theme_name = questions_tbl.theme_name
            #         LEFT JOIN strengths_tbl ON strengths_tbl.theme_name = themes_tbl.theme_name
            #         LEFT JOIN concerns_tbl ON concerns_tbl.theme_name = themes_tbl.theme_name
            #         WHERE questions_tbl.question = ?
            #     """, (question,))
            #     rows = cursor.fetchall()
            #     print(f"SQL pull Rows: {rows}")
            #     strengths = [row[0] for row in rows if row[0]]
            #     print(f"Strengths: {strengths}")
            #     concerns = [row[1] for row in rows if row[1]]
            #     print(f"Concerns: {concerns}")

                cursor.execute("""
                    SELECT themes_tbl.theme_name, strengths_tbl.strength, concerns_tbl.concern
                    FROM questions_tbl
                    JOIN themes_tbl ON themes_tbl.theme_name = questions_tbl.theme_name
                    LEFT JOIN strengths_tbl ON strengths_tbl.theme_name = themes_tbl.theme_name
                    LEFT JOIN concerns_tbl ON concerns_tbl.theme_name = themes_tbl.theme_name
                    WHERE questions_tbl.question = ?
                """, (question,))
                rows = cursor.fetchall()

                # Print the rows for debugging
                # print(f"SQL pull Rows: {rows}")

                # Extract data
                theme_name = rows[0][0] if rows else None  # First column is theme_name
                strengths = [row[1] for row in rows if row[1]]  # This loops through each ROW and Second column is strength,  only include if there is a value in present
                concerns = [row[2] for row in rows if row[2]]  # this loops through al lthe rows and extracts the Third column as concern

                # Step 5 -  Evaluate the answer using provided question, competency,  strength and concerns via  LLM model. Then print the result
                # st.session_state.evaluation_output = evaluate_answer(answer, strengths, concerns, question)
                st.session_state.evaluation_output = evaluate_answer(answer, theme_name, strengths, concerns, question)

                st.markdown(f"#### Evaluation for Question ({counter}):")
                st.markdown(f"({question})")
                # print(f"Questions list : {question}")
                st.write(st.session_state.evaluation_output)
                counter += 1

            conn.close()


# Allow ability to run it independentently
if __name__ == "__main__":
    interview_question_prep()



