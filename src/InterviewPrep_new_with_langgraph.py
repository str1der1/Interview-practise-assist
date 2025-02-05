
# System imports 
import os
import sqlite3
import streamlit as st
# from dotenv import load_dotenv
from pprint import pprint
from datetime import datetime
import random
# from tabulate import tabulate

# Langgraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import TypedDict, Dict, List, Set, Annotated, Sequence, TypeVar, Union
from typing_extensions import Annotated
# from langchain_openai import ChatOpenAI
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

# Imports for  STT and TTS
import pyttsx3
from gtts import gTTS
import tempfile
from elevenlabs.client import ElevenLabs
from elevenlabs import save


# Load environment variables
# load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
db_url = st.secrets["database_url"]
elevenlabs_api_key = st.secrets["ELEVENLABS_API_KEY"]

# Initialise OpenAI client for Whisper audio transcription 
openAI_client = OpenAI()

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.4,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialise the main dictionary to store the questions and answers
# st.session_state.stored_q_and_a_dict = {}  # Dictionary to store answers for each question asked , Key = Question,  Value = Answer
st.session_state.stored_q_and_a_list = []  # List of Dictionary elements to store 1/ questions,  2/ answers, 3/ evaluation

# Connect to SQLite Database
def connect_db():
    conn = sqlite3.connect("amazonQBank_values_and_questions.db")  # Replace with your DB file
    return conn

# Fetch competencies from the database
def fetch_competencies():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, theme_name, theme_description FROM themes_tbl")
    data = cursor.fetchall()
    conn.close()
    return data

# Fetch random questions for selected competencies from the database
def fetch_questions(competencies, stored_question_list, n=3 ):
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
    
    for q in questions:
        entry = {"question": q[0], "answer": "", "evaluation": ""}
        stored_question_list.append(entry)
   


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
    



# #####################################

# Define TypedDict for each evaluation type
class STAREval(TypedDict):
    star_feedback: str

class ClarityEval(TypedDict):
    clarity_feedback: str

class StrengthWeaknessEval(TypedDict):
    strength_weakness_feedback: str

# Define the overall state
class State(TypedDict):
    question: str
    answer: str
    theme_name: str
    strengths: List[str]
    concerns: List[str]
    star_feedback: str
    clarity_feedback: str
    strength_weakness_feedback: str
    final_feedback: str

def evaluate_star(state: State) -> STAREval:
    """Evaluates STAR principle in a given interview answer."""

    prompt = f"""
    You are an interview evaluator assessing STAR ( Situation , Task, Assessment, Result) in a given interview response. 
    
    # Role 
    - Your goal is to determine whether a given answer provides sufficient consideration with clear Situation clarity,  Task clarity,  Assesment clarity,  Result clarity, in the answer for the question.  

    # Input Data
    - Question: {state['question']}
    - Answer: {state['answer']}
    - Theme Name: {state['theme_name']}

    # Report
    - Provide Bullet points for both highlighting callouts on Situation, Task, Assessment , Result.  Tag each response at the start with feedback corresponding to "Situation -    " or "Task -    " or "Assessment  -   " or "Result  -  " where applicable
    - Tag each bullet point with "[Good]  " where confirmation of positive feedback or "[Improve]  " where answer needs improvement
    
    # Checks 
    - If the provided answer is too short or lacks enough words for meaningful evaluation, explicitly state this is the case. 
    - Do not infer missing details or generate an evaluation if information is insufficient.

    # Formatting :
    - Provide only concise, structured feedback in short sentences in bullet point format. 
    - Do not include unnecessary explanations or assumptions.
    - Do not use any header markdowns in feedback.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"star_feedback": response.content}




def evaluate_clarity(state: State) -> ClarityEval:
    """Evaluates Clarity and coherence in a given answer."""
    
    prompt = f"""
    You are an interview evaluator assessing "clarity and coherence" in a given interview response. 
    
    # Role 
    - Your goal is to determine whether a given answer provides sufficient clarity and coherence matching the answer and the question.  

    # Input Data
    - Question: {state['question']}
    - Answer: {state['answer']}

    # Report
    - Provide Bullet points for both clarity and coherence of the answer,  tag each response with either "[clarity]  " or "[coherence]  "
    - tag each bullet point with "[Good]  " where confirmation of positive feedback or "[Improve]  " where answer needs improvement

    # Checks 
    - If the provided answer is too short or lacks enough words for meaningful evaluation, explicitly state this is the case. 
    - Do not infer missing details or generate an evaluation if information is insufficient.

    # Formatting :
    - Provide only concise, structured feedback in short sentences in bullet point format. 
    - Do not include unnecessary explanations or assumptions.
    - limit feedback for each section to 3-4 bullet points.  
    - Do not use any header markdowns in feedback.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"clarity_feedback": response.content}




def evaluate_strength_weakness(state: State) -> StrengthWeaknessEval:
    """Checks if the answer is matching the strengths, or including the  weaknesses ."""
    prompt = f"""
    You are an interview evaluator assessing "strengths and weaknesses" of an interview response. 
    
    # Role 
    - Your goal is to determine whether a given answer matches to specific informed/given strengths or indicates any informed/given weaknesses.  

    # Input Data
    - Question: {state['question']}
    - Answer: {state['answer']}
    - Theme Name: {state['theme_name']}
    - Strength: {state['strengths']}
    - Weakness: {state['concerns']}

    # Report
    - Provide Bullet points for where answer is matching strengths or weakness in the answer, tag each response with either "[Strength]  " or "[Weak]  "
    - tag each bullet point with "[Good]  " where confirmation of positive feedback or "[Improve]  " where answer needs improvement

    # Formatting :
    - Provide only concise, structured feedback in short sentences in bullet point format. 
    - Do not include unnecessary explanations or assumptions.
    - limit feedback for each section to maximum of 3-4 bullet points.  
    - Do not use any header markdowns in feedback.

        # Checks 
    - If provided "strength criteria" is empty or too short( less than 10 words),  mention there is not enough information to measure strength
    - If provided "weakness criteria" is empty or too short ( less than 10 words),  mention there is not enough information to measure weakness
    - If the provided answer is too short or lacks enough words for meaningful evaluation, explicitly state this is the case. 
    - Do not infer missing details or generate an evaluation if information is insufficient.

    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"strength_weakness_feedback": response.content}




def aggregate_feedback(state: State) -> Dict:
    """Combine multiple evaluations into a final summary."""

    # final_summary = ""
    # Trial 1 - making a tool to just collect feedbacks together    
    # if state.get('star_feedback'):
    #     final_summary += f"\n**STAR Evaluation**:\n{state['star_feedback']}\n"
    # if state.get('clarity_feedback'):
    #     final_summary += f"\n**Clarity Evaluation**:\n{state['clarity_feedback']}\n"
    # if state.get('strength_weakness_feedback'):
    #     final_summary += f"\n**Strength and Weakness Evaluation**:\n{state['strength_weakness_feedback']}\n"

    # state["final_feedback"] = final_summary
    # # setattr(state, "final_feedback", final_summary)

    # return {
    #     "final_feedback": final_summary

    # Trial 2 -  using LLM as a orchestrator for feedbacks 
    prompt = f"""
    You are an interview evaluator collecting assesments from varioius tools for assessing aspects of an interview response. 
    
    # Role 
    - Your goal is to collect feedbacks from various assessors and collate them all into collective feedback mechanism for informing the interviewee

    # Input Data
    - STAR feedback: {state['star_feedback']}
    - Clarity and cohesiveness feedback: {state['clarity_feedback']}
    - Strength and Weakness feedback: {state['strength_weakness_feedback']}

    # Report
    - Provide Bullet points for each aspect area of the feedback.
    - Where tags exist,  keep those in
    - Where tags do not exist , make sure to indicate with by adding tage "[Good]  " where confirmation of positive feedback or "[Improve]  " where answer needs improvement

    # Formatting :
    - Make sure you have clear sections for each STAR feedback,  Clarity and cohesiveness feedback, Strength and Weakness feedback
    - Do not make any additional explanations or assumptions than what is provided.
    - limit feedback for each section to maximum of 3-4 bullet points.  
    - Do not use any header markdowns in feedback.

    # Checks 
    - If it is indicated in a section that not enough information is available or answer is too short.  Make this clear in the generated feedback 
    - Do not infer missing details or generate an evaluation if information is insufficient.

    """
    response = llm.invoke([HumanMessage(content=prompt)])


    state["final_feedback"] = response.content

    return {
        "final_feedback": response.content
    }


def text_to_speech(text, method="offline"):
    """
    Converts text to speech.
    - `method="offline"` uses pyttsx3 (no internet required)
    - `method="online"` uses gTTS (Google TTS, requires internet)
    """

    # DEBUG 
    # print(f"Text to speech method called with method: {method} and text: {text}")

    if not text:
        return None  # No text to speak
    
    if method == "offline":
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    
    elif method == "google":
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name  # Return file path for Streamlit playback

    elif method == "elevenlabs":
        client = ElevenLabs(api_key=elevenlabs_api_key)
        audio = client.generate(text=text, voice="Rachel")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        save(audio, temp_file.name)
        return temp_file.name  # Return file path for playback
    
    else:
        st.error("Invalid method for text-to-speech")
        print("Invalid method for text-to-speech")

    return None





# #####################################




# #######  MAIN function ######


# Initialize Graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("star_evaluation", evaluate_star)
workflow.add_node("clarity_evaluation", evaluate_clarity)
workflow.add_node("strength_weakness_evaluation", evaluate_strength_weakness)
workflow.add_node("aggregate_feedback", aggregate_feedback)

# Set entry point
# workflow.set_entry_point("star_evaluation")
workflow.add_edge(START, "star_evaluation")
workflow.add_edge(START, "clarity_evaluation")
workflow.add_edge(START, "strength_weakness_evaluation")

# Define the parallel execution flow
workflow.add_edge(["star_evaluation","clarity_evaluation", "strength_weakness_evaluation"], "aggregate_feedback")
workflow.add_edge("aggregate_feedback", END)

# Compile the workflow
graph = workflow.compile()

# Define graph invocation and execution function.  This function will be called in the main function.  This is being done after the graph is defined
def run_graph_evaluation(question: str, answer: str, theme_name: str, strengths: List[str], concerns: List[str]) -> str:
    """Prepare the initial state and invoke the LangGraph workflow."""
    initial_state = {
        "question": question,
        "answer": answer,
        "theme_name": theme_name,
        "strengths": strengths,
        "concerns": concerns,
        "star_feedback": "",
        "clarity_feedback": "",
        "strength_weakness_feedback": "",
        "final_feedback": ""
    }
    result = graph.invoke(initial_state)

    # Print the output each time graph is invoked 
    print ("\n\nGraph invoke called, The full object is printed out as of  Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pprint(result)

    # return result["final_feedback"]
    return result


# Main function 
def interview_question_evaluator2_with_langgraph():

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
        st.session_state.question_count  = st.slider("Number of Questions per competency", min_value=1, max_value=5, value=0)
        st.session_state.pull_questions = st.button("Pull Questions")


    # Step 3 - Pull questions from the database  - given competencies and number of questions,  pull X questions from the DB
    if "pull_questions" in st.session_state and st.session_state.question_count > 0 and st.session_state.pull_questions==True:

        # DEBUG
        # print(f" Pull questions is : {st.session_state.pull_questions}  and question count per competency is : {st.session_state.question_count}")

        # Isolate just the competency from the list selected 
        competency_ids = [comp[1] for comp in st.session_state.selected_competencies]
        # print(f"Competency_IDs: {competency_ids}")
        # Fetch questions (count per competency) from DB
        fetch_questions(competency_ids, st.session_state.stored_q_and_a_list, n=st.session_state.question_count)


    # Show Questions 
    #  Based on if there is stored Questions,  hopefully answers and evals too,  but dont know what state this is called 
    if "stored_q_and_a_list" in st.session_state:
        st.header("3. Questions")

        # Input box for Questions and answers 
        for i, questions_and_answers_listitem in enumerate(st.session_state.stored_q_and_a_list):
            # Pull the question from the stored list of Q & As
            question = questions_and_answers_listitem.get('question')

            # Show the question
            st.markdown(f"###### Q{i + 1}):\n{question}")
            
            # Provide input box for answer -  Check if answer already present , if yes, then show
            if questions_and_answers_listitem.get('answer'):
                st.session_state.stored_q_and_a_list[i]['answer'] = st.text_area(f"Your answer for Question {i + 1}:", value=questions_and_answers_listitem.get('answer'), height=100)
            else:
                st.session_state.stored_q_and_a_list[i]['answer'] = st.text_area(f"Your answer for Question {i + 1}:", height=100)
                
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
                st.session_state.stored_q_and_a_list[i]['answer']  = transcription

                st.rerun()  # Force a refresh to update the the text area population


    # Evaluation loop 
    #  Step 4: Evaluate Answers   -  Two parts 
    #               1) pull list of strengths and concerns for each question competency 
    #               2) run the langgraph tool to evaluate
    st.divider()

    # Show Questions 
    #  Based on if there is stored Questions,  hopefully answers and evals too,  but dont know what state this is called 
    if "stored_q_and_a_list" in st.session_state:
        # show Evaluation option if there is stored questions and answers
        st.header("4. Evaluate")
        if st.button("Run Evaluation on Answers"):
            st.header("5. Results")

            conn = connect_db()
            cursor = conn.cursor()

            # Loop therough each question and answer in the stored list
            for i, questions_and_answers_listitem in enumerate(st.session_state.stored_q_and_a_list):
                # Pull the question from the stored list of Q & As
                question = questions_and_answers_listitem.get('question')
                answer = questions_and_answers_listitem.get('answer')

                # Execute the SQL for specific question to pull strengths and concerns
                cursor.execute(
                    """
                    SELECT DISTINCT 
                        themes.theme_name, 
                        strengths.strength, 
                        concerns.concern
                    FROM questions_tbl
                    JOIN (
                        SELECT DISTINCT theme_name 
                        FROM themes_tbl
                    ) AS themes ON themes.theme_name = questions_tbl.theme_name
                    LEFT JOIN (
                        SELECT DISTINCT theme_name, strength 
                        FROM strengths_tbl
                    ) AS strengths ON strengths.theme_name = themes.theme_name
                    LEFT JOIN (
                        SELECT DISTINCT theme_name, concern 
                        FROM concerns_tbl
                    ) AS concerns ON concerns.theme_name = themes.theme_name
                    WHERE questions_tbl.question = ?
                    """, (question,))

                rows = cursor.fetchall()

                # DEBUG : 
                # Print pulled SQL rows for debugging
                # print(f"SQL pull Rows: {rows}")
                # print(tabulate(rows, headers=["Theme", "Strength", "Concern"], tablefmt="grid"))

                # Extract data ,  make sure Strenghts and Concerns is unique and sorted
                theme_name = rows[0][0] if rows else None  # First column is theme_name
                # Use sets Type for automatic uniqueness handling
                strengths_set: Set[str] = {row[1] for row in rows if row[1] is not None}
                concerns_set: Set[str] = {row[2] for row in rows if row[2] is not None}
                # Convert sets Into sorted lists for consistent ordering
                strengths = sorted(list(strengths_set))
                concerns = sorted(list(concerns_set))

                # Step 5 -  Evaluate the answer using provided question, competency,  strength and concerns via  LLM model. Then print the result
                with st.spinner(f"Running evaluation for Question {i+1} ..."):
                    # Main execution of langgraph , Return the full class instance , not just the final feedback from langgraph
                    st.session_state.evaluation_output = run_graph_evaluation(question=question, answer=answer, theme_name=theme_name, strengths=strengths, concerns=concerns)

                    questions_and_answers_listitem['evaluation'] = st.session_state.evaluation_output['final_feedback']

                    # Present this 
                    st.markdown(f"#### Evaluation for Question ({i+1}):")
                    st.write(f"**Q :**  {question}")
                    st.write(f"**A :**  {answer}")
                    st.markdown(questions_and_answers_listitem.get('evaluation'))

                # Divider betweeen the evaluation responses
                st.divider()

            conn.close()
    
    st.divider()


    # Step 6 - Add ability to speak out the feedback for each question and answer 
    st.session_state.speak_feedback = st.button("Speak Feedback")
    if st.session_state.speak_feedback and "stored_q_and_a_list" in st.session_state:

        print("####  Speak Feedback button clicked and stored data present")

        # Loop therough each question and answer in the stored list
        for i, questions_and_answers_listitem in enumerate(st.session_state.stored_q_and_a_list):

            if "evaluation" not in questions_and_answers_listitem:
                st.error(f"Q{i+1} - Missing evaluation data ")
            else:
                st.write(f"For Question {i+1}:")
                
                # DEBUG
                # print(f"pre call the audio spinnner,  Question {i+1} ")

                # Generate and play audio feedback
                with st.spinner('Generating audio feedback...'):
                    # Change to "offline" for pyttsx3, "google" for gTTS,  "elevenlabs" for ElevenLabs
                    audio_file = text_to_speech(questions_and_answers_listitem['evaluation'],  method="google")

                if audio_file:
                    st.audio(audio_file, format="audio/mp3")


    # Step 6 - Add ability to add more questions 
    # if st.button("Add More Questions"):
    #     st.header("5. Add More Questions")


    # DEBUG - print out the full list of stored answers and Questions 
    if 'stored_q_and_a_list' in st.session_state:
        print("Stored Questions and Answers in streamlit saved memory state: ")
        print(f" Number of Questions is :  {len(st.session_state['stored_q_and_a_list'])}")
        pprint(st.session_state['stored_q_and_a_list'])


# Allow ability to run it independentently
if __name__ == "__main__":
    interview_question_evaluator2_with_langgraph()



