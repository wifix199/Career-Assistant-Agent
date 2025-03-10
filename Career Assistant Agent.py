# GenAI Career Assistant Agent
# Requirements: Install packages first
# pip install langchain==0.3.7 langchain-community==0.3.7 langchain-google-genai==2.0.4 duckduckgo_search==6.3.4 langgraph==0.2.48 python-dotenv

from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from IPython.display import display, Image, Markdown
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
# Import MessagesPlaceholder from langchain.memory instead of langchain.chains.memory
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchResults
from datetime import datetime

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = "Your_openai_api_key" # Replace with your actual API key

# Define LLM
llm = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0.5)

class State(TypedDict):
    query: str
    category: str
    response: str

def trim_conversation(prompt):
    max_messages = 10
    return trim_messages(
        prompt,
        max_tokens=max_messages,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

def save_file(data, filename):
    folder_name = "Agent_output"
    os.makedirs(folder_name, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename}_{timestamp}.md"
    file_path = os.path.join(folder_name, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)
    return file_path

def show_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    display(Markdown(content))

class LearningResourceAgent:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def TutorialAgent(self, user_input):
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        response = agent_executor.invoke({"input": user_input})
        path = save_file(str(response.get('output')).replace("```markdown", "").strip(), 'Tutorial')
        print(f"Tutorial saved to {path}")
        return path

    def QueryBot(self, user_input):
        record_QA_session = []
        record_QA_session.append('User Query: %s \n' % user_input)
        self.prompt.append(HumanMessage(content=user_input))
        while True:
            self.prompt = trim_conversation(self.prompt)
            response = self.model.invoke(self.prompt)
            record_QA_session.append('\nExpert Response: %s \n' % response.content)
            self.prompt.append(AIMessage(content=response.content))
            print('*'*50 + 'AGENT' + '*'*50)
            print("EXPERT AGENT RESPONSE:", response.content)
            print('*'*50 + 'USER' + '*'*50)
            user_input = input("\nYOUR QUERY: ")
            record_QA_session.append('\nUser Query: %s \n' % user_input)
            self.prompt.append(HumanMessage(content=user_input))
            if user_input.lower() == "exit":
                path = save_file(''.join(record_QA_session), 'Q&A_Doubt_Session')
                print(f"Q&A Session saved to {path}")
                return path

class InterviewAgent:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def Interview_questions(self, user_input):
        chat_history = []
        questions_bank = ''
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        while True:
            if user_input.lower() == "exit":
                break
            response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            questions_bank += str(response.get('output')).replace("```markdown", "").strip() + "\n"
            chat_history.extend([HumanMessage(content=user_input), response["output"]])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            user_input = input("You: ")
        path = save_file(questions_bank, 'Interview_questions')
        return path

    def Mock_Interview(self):
        interview_record = []
        interview_record.append('Candidate: I am ready for the interview.\n')
        self.prompt.append(HumanMessage(content='I am ready for the interview.'))
        while True:
            self.prompt = trim_conversation(self.prompt)
            response = self.model.invoke(self.prompt)
            self.prompt.append(AIMessage(content=response.content))
            print("\nInterviewer:", response.content)
            interview_record.append('\nInterviewer: %s \n' % response.content)
            user_input = input("\nCandidate: ")
            interview_record.append('\nCandidate: %s \n' % user_input)
            self.prompt.append(HumanMessage(content=user_input))
            if user_input.lower() == "exit":
                path = save_file(''.join(interview_record), 'Mock_Interview')
                return path

class ResumeMaker:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def Create_Resume(self, user_input):
        chat_history = []
        while True:
            if user_input.lower() == "exit":
                break
            response = self.agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            chat_history.extend([HumanMessage(content=user_input), response["output"]])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            user_input = input("You: ")
        path = save_file(str(response.get('output')).replace("```markdown", "").strip(), 'Resume')
        show_md_file(path)
        return {"response": path}

class JobSearch:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = prompt
        self.tools = DuckDuckGoSearchResults()

    def find_jobs(self, user_input):
        results = self.tools.invoke(user_input)
        chain = self.prompt | self.model
        jobs = chain.invoke({"result": results}).content
        path = save_file(str(jobs).replace("```markdown", "").strip(), 'Job_search')
        show_md_file(path)
        return {"response": path}

def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n"
        "Give the number only as an output.\n\n"
        "Examples:\n"
        "1. Query: 'What are the basics of generative AI...' -> 1\n"
        "2. Query: 'Can you help me improve my resume...' -> 2\n"
        "3. Query: 'What are some common questions...' -> 3\n"
        "4. Query: 'Are there any job openings...' -> 4\n\n"
        "Now, categorize the following customer query:\n"
        "Query: {query}"
    )
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def handle_learning_resource(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the user query into Tutorial or Question...\n"
        "Categories:\n"
        "- Tutorial: Creating tutorials/blogs\n"
        "- Question: General queries about GenAI\n"
        "Examples...\n"
        "Now, categorize: The user query is: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def handle_interview_preparation(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize into Mock or Question...\n"
        "Categories:\n"
        "- Mock: Mock interviews\n"
        "- Question: Interview topic questions\n"
        "Examples...\n"
        "Now, categorize: The user query is: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def route_query(state: State):
    if '1' in state["category"]:
        return "handle_learning_resource"
    elif '2' in state["category"]:
        return "handle_resume_making"
    elif '3' in state["category"]:
        return "handle_interview_preparation"
    elif '4' in state["category"]:
        return "job_search"
    else:
        print("Please ask a relevant question.")
        return False

def route_interview(state: State) -> str:
    if 'Question'.lower() in state["category"].lower():
        return "interview_topics_questions"
    elif 'Mock'.lower() in state["category"].lower():
        return "mock_interview"
    else:
        return "mock_interview"

def route_learning(state: State):
    if 'Question'.lower() in state["category"].lower():
        return "ask_query_bot"
    elif 'Tutorial'.lower() in state["category"].lower():
        return "tutorial_agent"
    else:
        return False

def tutorial_agent(state: State) -> State:
    system_message = """You are a Senior GenAI Developer and blogger..."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.TutorialAgent(state["query"])
    show_md_file(path)
    return {"response": path}

def ask_query_bot(state: State) -> State:
    system_message = """You are an expert GenAI Engineer providing solutions..."""
    prompt = [SystemMessage(content=system_message)]
    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.QueryBot(state["query"])
    show_md_file(path)
    return {"response": path}

def interview_topics_questions(state: State) -> State:
    system_message = """You are a researcher providing interview questions..."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Interview_questions(state["query"])
    show_md_file(path)
    return {"response": path}

def mock_interview(state: State) -> State:
    system_message = """You are a GenAI Interviewer conducting mock interviews..."""
    prompt = [SystemMessage(content=system_message)]
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Mock_Interview()
    show_md_file(path)
    return {"response": path}

def handle_resume_making(state: State) -> State:
    system_message = """You are a resume expert for tech roles..."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    resumeMaker = ResumeMaker(prompt)
    path = resumeMaker.Create_Resume(state["query"])
    show_md_file(path)
    return {"response": path}

    # Check if the user wants to modify an existing resume
    modify_existing = input("Do you want to modify an existing resume? (yes/no): ")
    if modify_existing.lower() == "yes":
        # Get the resume file path from the user
        resume_file_path = input("Enter the path to your resume file: ")
        # Load the resume text from the file
        with open(resume_file_path, "r") as f:
            resume_text = f.read()
        # Call Modify_Resume
        path = resumeMaker.Modify_Resume(resume_text)

def job_search(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Refactor job search results into a .md file..."
    )
    jobSearch = JobSearch(prompt)
    state["query"] = input('Enter job location and roles:\n')
    path = jobSearch.find_jobs(state["query"])
    show_md_file(path)
    return {"response": path}

# Workflow Setup
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("handle_learning_resource", handle_learning_resource)
workflow.add_node("handle_resume_making", handle_resume_making)
workflow.add_node("handle_interview_preparation", handle_interview_preparation)
workflow.add_node("job_search", job_search)
workflow.add_node("mock_interview", mock_interview)
workflow.add_node("interview_topics_questions", interview_topics_questions)
workflow.add_node("tutorial_agent", tutorial_agent)
workflow.add_node("ask_query_bot", ask_query_bot)

workflow.add_edge(START, "categorize")
workflow.add_conditional_edges("categorize", route_query, {
    "handle_learning_resource": "handle_learning_resource",
    "handle_resume_making": "handle_resume_making",
    "handle_interview_preparation": "handle_interview_preparation",
    "job_search": "job_search"
})
workflow.add_conditional_edges("handle_interview_preparation", route_interview, {
    "mock_interview": "mock_interview",
    "interview_topics_questions": "interview_topics_questions",
})
workflow.add_conditional_edges("handle_learning_resource", route_learning, {
    "tutorial_agent": "tutorial_agent",
    "ask_query_bot": "ask_query_bot",
})

workflow.add_edge("handle_resume_making", END)
workflow.add_edge("job_search", END)
workflow.add_edge("interview_topics_questions", END)
workflow.add_edge("mock_interview", END)
workflow.add_edge("ask_query_bot", END)
workflow.add_edge("tutorial_agent", END)

workflow.set_entry_point("categorize")
app = workflow.compile()

def run_user_query(query: str) -> Dict[str, str]:
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "response": results["response"]
    }

if __name__ == "__main__":
    # Test cases
    query = "I want to learn Langchain and langgraph. Create a tutorial."
    result = run_user_query(query)
    print(result)

    query = "I want to update my resume for a GenAI role. Here's my current resume: [paste resume text or file path]"
    result = run_user_query(query)
    print(result)

    query = "Can you help me modify my resume for a GenAI job"
    result = run_user_query(query)
    print(result)

    query = "Find GenAI jobs in USA"
    result = run_user_query(query)
    print(result)