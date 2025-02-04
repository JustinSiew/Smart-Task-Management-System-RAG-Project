# Install necessary libraries
#pip install --upgrade gradio
#pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
#pip install pypdf
#pip install fpdf
#pip install PyPDF2

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import gradio as gr
from fpdf import FPDF

# Step 1: Define the main class for handling the chat extraction
class ChatExtractor:
    def __init__(self):
        """Initialize class variables and set API keys."""
        self.pdf_paths = []
        self.vectorstore = None
        self.all_documents = []
        self.summary_documents = []
        self.todo_documents = []

        # Set OpenAI API Key
        os.environ["OPENAI_API_KEY"] = "sk-k343y-_bcqi5gCfODKACE5GAppoaGaFEWOhF8fwC70T3BlbkFJJY8W17JYH-lTG5sK3Xuf5ZNpX57QtV4sW32ImM-JMA"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    # Step 2: File processing methods
    def process_uploaded_files(self, files):
        """Process uploaded PDF files, split documents, and embed content."""
        self.pdf_paths = [file.name for file in files]
        try:
            self.all_documents = self.load_chat_history(self.pdf_paths)
            chunks = self.split_documents(self.all_documents)
            self.vectorstore = self.embed_documents(chunks)
            self.generate_summary_and_todo_pdfs(self.pdf_paths)
            return "Files uploaded and processed successfully!"
        except Exception as e:
            return f"Error processing files: {str(e)}"

    def load_chat_history(self, filepaths):
        """Load chat history from PDF files."""
        all_documents = []
        for filepath in filepaths:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            all_documents.extend(documents)
        return all_documents




    # Step 3: Document splitting and embedding methods
    def split_documents(self, documents):
        """Split large documents into smaller chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", "!", "?", ","]
        )
        return text_splitter.split_documents(documents)

    def embed_documents(self, chunks):
        """Embed document chunks using OpenAI embeddings and Chroma."""
        embeddings = OpenAIEmbeddings()
        return Chroma.from_documents(documents=chunks, embedding=embeddings)


    # Step 4: Summary and To-Do PDF generation methods
    def generate_summary_and_todo_pdfs(self, filepaths):
        """Generate summary and to-do PDFs and update the vector store."""
        summary_pdf = FPDF()
        summary_pdf.add_page()
        summary_pdf.set_font("Arial", size=12)

        todo_pdf = FPDF()
        todo_pdf.add_page()
        todo_pdf.set_font("Arial", size=12)

        for filepath in filepaths:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            full_content = "\n\n".join(doc.page_content for doc in documents)

            # Summarize content and add to summary PDF
            summary = self.summarize_content(full_content)
            summary_pdf.multi_cell(0, 10, summary)

            # Generate to-do list and add to to-do PDF
            todo_list = self.generate_todo_list(summary)
            previous_line_was_task = False
            for line in todo_list.split("\n"):
                if line.strip() and not any(keyword in line for keyword in ["Not specified", "Not Started"]):
                    if line.startswith("Task:"):
                        if previous_line_was_task:
                            todo_pdf.ln(10)
                        previous_line_was_task = True
                    else:
                        previous_line_was_task = False
                    todo_pdf.multi_cell(0, 10, line)

        # Save PDFs
        summary_pdf.output("summary.pdf")
        todo_pdf.output("todo_list.pdf")

        # Reload documents into the vectorstore
        self.summary_documents = self.load_chat_history(["summary.pdf"])
        self.todo_documents = self.load_chat_history(["todo_list.pdf"])

        all_documents = self.all_documents + self.summary_documents + self.todo_documents
        chunks = self.split_documents(all_documents)
        self.vectorstore = self.embed_documents(chunks)

    # Step 5: OpenAI interaction methods
    def summarize_content(self, content):
        """Use OpenAI model to summarize chat history."""
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        prompt = """
        
        You are an expert at summarizing chat history. Your task is to generate a comprehensive and concise summary of the provided chat history from a PDF document. Follow these rules:
        
        1. Extract Key Details:
           - Identify all the people involved (e.g., names).
           - Identify all commitment (e.g., "having lunch" , "meet up", "visit").
           - Identify all times and dates (e.g., "tomorrow at 1 PM") using context, ensuring that relative dates (like “next Friday,” “tomorrow,” or specific weekdays like “Saturday”) are converted accurately to the exact calendar date. For example, if today is 30 January 2025, and someone mentions meeting on “Friday,” the correct date would be 31 January 2025.
           - Identify all locations (e.g., "new restaurant").
    
        2. Write the Summary:
           - Use 1-2 sentences for each chat or event.
           - Ensure the summary is short and to the point.
           - Include all key details (who, what, when, where) for each chat or event.
           - For dates: Convert any relative reference to the exact calendar date by considering the current date and context. For example, if someone mentions "next week" and today is 30 January 2025, "next week" should be interpreted as the week starting 6 February 2025, not the week starting 30 January 2025.
        
        3. Example Format:
           - "[Person] wants to [action] at [time] on [date] at [location]."
           - If the event is on a relative day (like “Friday”), ensure the model derives the exact date (e.g., "meeting on Friday, 31 January 2025”).
           
        Extraction Example:
        Chat 3: Project Collaboration
        User: Alex Rodriguez  
        Team Member: Emma Wilson  
        Date: December 30, 2024
        Alex: Hi Emma, could you review the latest design mockups I sent?
        Emma: Morning Alex! Yes, I'll take a look at them right away.
        Alex: Thanks! Pay special attention to the new navigation layout.
        Emma: Will do. I noticed you added some interesting micro-interactions. Let's discuss those during our team meeting tomorrow.
        
        Analysis:
        Based on the conversation, the team meeting mentioned by Emma is scheduled for "tomorrow," which refers to December 31, 2024.
        So, the  date for discussing the micro-interactions in the team meeting is December 31, 2024.
        
        Summarize text:
        -Alex Rodriguez asks Emma Wilson to review design mockups on December 31, 2024, with a team meeting scheduled.
        
        Input Chat History: 
        {chat_history}
        Your Task: Generate a short summary of the chat history using the format and rules above. Make sure to derive dates accurately based on the context, including cross-referencing relative date mentions (like “tomorrow,” “next week,” or specific weekdays like “Friday”) to get the precise date.

        """.format(chat_history=content)
        response = llm(prompt)
        return response if isinstance(response, str) else getattr(response, 'content', str(response))

    def extract_todo_list_from_pdf(self):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader("todo_list.pdf")
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            return full_text
        except Exception as e:
             return f"Error reading todo_list.pdf: {str(e)}"

    def format_todo_list(self, content):
        lines = content.split('\n')
        formatted_lines = []
        for line in lines:
            if line.strip() and not any(keyword in line for keyword in ["Not specified", "Not Started"]):
                if line.strip().startswith("Task:") or line.strip().startswith("Current task list:"):
                    formatted_lines.append("")
                formatted_lines.append(line.strip())
        return "\n".join(formatted_lines)


    def generate_todo_list(self, content=None):
        """Use OpenAI model to generate a detailed to-do list."""
        try:
            if content is not None:
                llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
                messages = [
                    {"role": "system", "content": "You are a task management expert analyzing chat history to create a comprehensive todo list."},
                    {"role": "user", "content": todo_prompt_template.format(retrieved_text=content, query="Generate detailed to do list")}
                ]
                response = llm(messages)
                todo_list = response.content if hasattr(response, 'content') else str(response)

                formatted_todo_list = self.format_todo_list(todo_list)

                todo_pdf = FPDF()
                todo_pdf.add_page()
                todo_pdf.set_font("Arial", size=12)
                for line in formatted_todo_list.split("\n"):
                    if line.strip():
                        if line.startswith("Task:"):
                            todo_pdf.ln(10)
                        todo_pdf.multi_cell(0, 10, line)
                todo_pdf.output("todo_list.pdf")

                return formatted_todo_list

            elif os.path.exists("todo_list.pdf"):
                return self.extract_todo_list_from_pdf()

            else:
                if os.path.exists("summary.pdf"):
                    summary_content = "\n\n".join(doc.page_content for doc in self.summary_documents)
                    return self.generate_todo_list(summary_content)
                else:
                    return "No todo list or summary available. Please upload PDF files first."

        except Exception as e:
            return f"Error generating todo list: {str(e)}"

    # Step 6: Query handling methods
    def handle_general_query(self, query):
        """Retrieve relevant documents and answer general queries."""
        retriever = self.vectorstore.as_retriever()
        retriever.search_kwargs['k'] = 50
        relevant_docs = retriever.get_relevant_documents(query)
        retrieved_text = "\n\n".join(doc.page_content for doc in relevant_docs)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        prompt = general_prompt_template.format(retrieved_text=retrieved_text, query=query)
        response = llm(prompt)

        if "No detail found." in response:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            extra_prompt = """
            You are a knowledgeable assistant. Answer the user's query based on your own knowledge.

            Query:
            {query}

            Your Task:
            Answer the query using your own knowledge.
            """.format(query=query)
            response = llm(extra_prompt)

        return response if isinstance(response, str) else getattr(response, 'content', str(response))

    def rag_workflow(self, query, prompt_type='general'):
        """Handle Retrieval-Augmented Generation (RAG) workflow."""
        if not self.vectorstore:
            return "Please upload PDF files first."

        try:
            if prompt_type == 'todo':
                formatted_todo_list = self.extract_todo_list_from_pdf()
                return self.format_todo_list(formatted_todo_list)

            elif prompt_type == 'summary':
                summary_content = "\n\n".join(doc.page_content for doc in self.summary_documents)
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                prompt = summary_prompt_template.format(retrieved_text=summary_content, query=query)
                response = llm(prompt)
                return response if isinstance(response, str) else getattr(response, 'content', str(response))

            else:
                return self.handle_general_query(query)

        except Exception as e:
            return f"Error in RAG workflow: {str(e)}"

# Step 7: Prompt templates
general_prompt_template = (
    "Chat history to analyze:\n{retrieved_text}\n\n"
    "Query: {query}\n\n"
    "You are now a helpful secretary of a Chat history. Answer the query based on the chat history. "
    "Make use of the information in the chat history. "
    "If the person and incident do not match, just say 'No detail found.' "
    "If no detail is found, respond using your own knowledge."
)

summary_prompt_template = (
    "Chat history to analyze:\n{retrieved_text}\n\n"
    "Query: {query}\n\n"
    "You are an expert at summarizing chat history. Your task is to generate a comprehensive and concise summary of the provided chat history. Follow these rules:\n"
    "1. Extract all key details (who, what, when, where).\n"
    "2. Write the summary in 1-2 sentences for each chat or incident.\n"
    "3. Ensure the summary is short and to the point.\n"
    "If no detail is found, respond using your own knowledge."
)

todo_prompt_template = '''
"Chat history to analyze:\n{retrieved_text}\n\n"
"Query: {query}\n\n"
"You are a highly detail-oriented secretary. Your task is to:\n"
"1. Carefully analyze ALL the provided chat history\n"
"2. Extract task, commitment, or responsibility mentioned\n"
"3. Include both explicit tasks and implied responsibilities\n"
"4. Format each task with complete details that are available\n\n"
"The \n{retrieved_text}\n\n can be the format : [Person] wants to [action] at [time] on [due date] at [location]\n "
"Generate a comprehensive to-do list in this format:\n"
"Current task list:\n"
Task: [Task description]
Priority: [High/Medium/Low based on urgency and importance]
Due Date: [Extract the specific date or day if mentioned, e.g., "Tomorrow" , "This Weekend" , "2 January 2025" ,"28 December 2024" , use 'Not specified' if unclear]
Time: [Extract specific time if mentioned, e.g., "1 PM", "Between 2-5PM", use 'Not specified' if unclear]
Location: [Extract specific location if mentioned, e.g., "New Restaurant", use 'Not specified' if unclear]
Workload: [Large/Medium/Small based on complexity and time required]
Client: [Who requested or benefits from the task]
Status: [Not Started/In Progress/Pending if status is mentioned]
'''



def create_interface():
    extractor = ChatExtractor()  # Ensure this class is correctly implemented elsewhere

    def process_message(message, history):
        if not message.strip():
            return history + [("user", message), ("assistant", "Please provide a valid input.")]

        response = extractor.rag_workflow(message, prompt_type="general")
        if "No detail found." in response:
            response = extractor.handle_general_query(message)

        history.append(("user", message))
        history.append(("assistant", response))
        return history

    def generate_todo_list(history):
        response = extractor.rag_workflow(
            "Generate detailed to-do list for me based on all the chat history",
            prompt_type="todo"
        )
        history.append(("assistant", response))
        return history

    def handle_file_upload(files, history):
        response = extractor.process_uploaded_files(files)
        history.append(("assistant", response))
        return history

    # Build the Gradio interface
    with gr.Blocks() as iface:
        gr.Markdown("# Chat Task Extractor")
        gr.Markdown("Upload your PDF files and tell me about your tasks.")

        chatbot = gr.Chatbot(
            value=[("assistant", "Hello! I'm your task management assistant. Please upload your PDF files to get started.")]
        )

        with gr.Row():
            file_upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF Files")
            todo_button = gr.Button("Generate To-Do List")

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Tell me about your tasks...",
                lines=2
            )
            send_button = gr.Button("Send")

        # Event bindings
        file_upload.upload(
            fn=handle_file_upload,
            inputs=[file_upload, chatbot],
            outputs=chatbot
        )

        send_button.click(
            fn=process_message,
            inputs=[msg, chatbot],
            outputs=chatbot
        )

        msg.submit(
            fn=process_message,
            inputs=[msg, chatbot],
            outputs=chatbot
        )

        todo_button.click(
            fn=generate_todo_list,
            inputs=chatbot,
            outputs=chatbot
        )

    return iface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, inbrowser=True)


