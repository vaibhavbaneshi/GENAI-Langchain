import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

##Set-up the Streamlit app
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2") 

with st.sidebar:
    groq_api_key=st.sidebar.text_input(label="Enter your Groq API Key", type="password")
    llm = ChatGroq(model="Gemma2-9b-It", api_key=groq_api_key)
    if st.button("Test API Key"):
        if not groq_api_key.strip():
            st.error("Please enter a valid api key.")
        else:
            try:
                result = llm.invoke("Say Hello!")
                st.success(result.content)
                st.success("Model Used: " + result.response_metadata['model_name'])
            except Exception as e:
                st.exception(f"Exception {e}")

##Initialize tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"
)

##Initialize Math Tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

## Create Prompt
prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=['question'],
    template=prompt
)

chain=LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool=Tool(
    name='Reasoning Tool',
    func=chain.run,
    description='A tool for answering logic-based and reasoning questions.'
)

##Initialize Agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant", "content": "I'm a Math chatbot who can answer all you math qusetions."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

##Lets start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response=assistant_agent(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant", "content":response})
            st.write("### Response")
            st.success("Result: " + response["output"])

    else:
        st.warning("Please enter the question")