from typing import TypedDict,Literal,Annotated,Dict,Any,Union
from pydantic import BaseModel,Field
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

class state(TypedDict):
    input_email:str
    email_classifer:str
    next:str
    email_info:list| None
    output_email_subject:str
    output_email_body:str


from dotenv import load_dotenv
load_dotenv()


llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.5

)

class email_classifer(BaseModel):
    email_type:Literal["type A","type B"]=Field(description="Classify the emails into type A,type B,")

llm_classifer=llm.with_structured_output(email_classifer)

def classifer(state:state):
    email_msg=state['input_email']
    prompt = f"""
            You are an expert email classifier for an e-commerce company. 
            Your task is to classify incoming emails into one of the following categories:

            - type A: Emails related to customer service topics, including:
                - Return/refund policies and processing
                - Product issues or complaints
                - Exchanges
                
                - Product availability or sizing questions
                - Account-related issues (login, password, profile)
                - General customer support inquiries

            - type B: Emails related to order lifecycle and delivery, including:
                - Order confirmation and status
                - All payment related quation
                - Tracking and shipping details
                - Estimated delivery time
                - Delayed or lost shipments
                - Billing, payments, or invoice inquiries
                - Cancelation or modification of orders
                - refund payment confirmation and status
                - Delivery instructions or issues (e.g., address not found)

            Instructions:
            - Carefully analyze the emails content and intent.
            - Do not assume or hallucinate information. Base classification only on the available content.
            - Return only the category label ("type A","type B") without explanation or extra text.

            Classify the following email: {email_msg}
            """
    response=llm_classifer.invoke(prompt)
    
    return {"email_classifer":response.email_type}

def router(state:state):
    email_type=state.get("email_classifer")
    if email_type=="type A":
        return {"next":"type A"}
    elif email_type=="type B":
        return {"next":"type B"}
  
class type_A(BaseModel):
    Subject:str
    body:str
    
type_A_llm=llm.with_structured_output(type_A)

def type_A_writer(state:state):
    email_info=state["email_info"]
    
    from langchain.vectorstores import Chroma,FAISS
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

    from dotenv import  load_dotenv
    load_dotenv()

    embadding=GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    DB_FAISS_PATH = 'vectorstore/db_faiss'

    db_loaded = FAISS.load_local(DB_FAISS_PATH, embadding, allow_dangerous_deserialization=True)
    retriever = db_loaded.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    query = state["input_email"]
    results = retriever.invoke(query)

    page_contents = [doc.page_content for doc in results]


    prompt=f"""
        You are an AI email assistant representing Click&Cart, responding on behalf of Swapnil Patil, Manager at Click&Cart.

        Generate professional email replies based on provided context and user queries. Each response must be concise, helpful, and maintain brand consistency.

        - Context: {page_contents} - Background information and relevant details
        - user email info :{email_info}

        Output Requirements

        Email Structure
        1. Professional Greeting
        - Use appropriate salutation (Dear [Name], Hello [Name], etc.)
        - Default to "Dear Customer" if name is not available

        2. Concise Body
        - Maximum 3-4 sentences
        - Address the customer's specific concern directly
        - Reference relevant information from the provided context
        - Maintain a helpful and professional tone

        3. Professional Closing
        - Use standard business closings (Best regards, Sincerely, etc.)
        - Include signature block

        Mandatory Elements
        - Company Representation: All emails sent on behalf of Click&Cart
        - Sender Identity: Swapnil Patil, Manager at Click&Cart
        - Standard Footer: Include this text exactly as written at the end of every email:
        
        If your problem is not solved and you are still facing issues, contact us.
        Fill this form: https://forms.gle/MYmJbdixXDQQ2iqx5
        

        Example Output Structure

        Dear [Customer Name],

        [2-3 sentences addressing their specific concern based on context]

        [1 sentence with next steps or resolution]

        Best regards,
        Swapnil Patil
        Manager, Click&Cart

        If your problem is not solved and you are still facing issues, contact us.
        Fill this form: https://forms.gle/MYmJbdixXDQQ2iqx5
         """
    response=type_A_llm.invoke(prompt)
    return {"output_email_subject":response.Subject,"output_email_body":response.body}

    
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///ecommerce_workspace.db")

from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()


system_prompt = """

 You are an AI email assistant representing Click&Cart, responding on behalf of Swapnil Patil, Manager at Click&Cart.
Given the following customer email , determine if you need to query the database to fetch relevent data use
use the SQLDatabaseToolkit to fetch the relevant data

If no DB call is needed, just write the response.

if you need to query the database to fetch the relevant data and 
email has not given any relevant data to query data based  write response to get that relevent data (like user id, product id , transaction id)

Finally, write a polite, clear, natural language email reply with the information retrieved.

Mandatory Elements
        - Company Representation: All emails sent on behalf of Click&Cart
        - Sender Identity: Swapnil Patil, Manager at Click&Cart
        - Standard Footer: Include this text exactly as written at the end of every email:
            If your problem is not solved and you are still facing issues, contact us.
            Fill this form: https://forms.gle/MYmJbdixXDQQ2iqx5



Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
"""

class type_B_val(BaseModel):
    subject:str
    body:str

def type_B_writer(state:state):
    email_txt=state["input_email"]

    prompt_template = system_prompt
    system_message = prompt_template.format(dialect="SQLite", top_k=5)

    # Create agent
    agent_executor = create_react_agent(
        llm, toolkit.get_tools(), state_modifier=system_message
    )


    events = agent_executor.stream(
        {"messages": [("user", email_txt)]},
        stream_mode="values",
    )

    final_response = None
    for event in events:
        event["messages"][-1].pretty_print()
        final_response = event["messages"][-1].content

    # Apply structured output parsing to the final response
    structured_llm = llm.with_structured_output(type_B_val)
    structured_result = structured_llm.invoke(f"Extract the email subject and body from this response: {final_response}")
  
    return {"output_email_subject":structured_result.subject,"output_email_body":structured_result.body}
    
import email_management
from email.header import decode_header
import imaplib
import email
import smtplib
import ssl
from email.message import EmailMessage
import os
import re
from datetime import datetime
import time

def get_emails_batch():
    """Fetch all unread emails and return them as a list"""
    email_address = 'mailreplyer5090@gmail.com'
    email_password = 'wfkj ohob ezhi arux'
    unread_emails = []
   
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(email_address, email_password)
        imap.select("INBOX")
        _, message_numbers = imap.search(None, "UNSEEN")
        
        if not message_numbers[0]:
            print("No unread emails found.")
            return []
        
        for num in message_numbers[0].split():
            _, msg = imap.fetch(num, "(RFC822)")
            email_body = msg[0][1]
            email_message = email.message_from_bytes(email_body)
            
            subject = decode_header(email_message["Subject"])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
                
            from_ = decode_header(email_message.get("From"))[0][0]
            if isinstance(from_, bytes):
                from_ = from_.decode()
            
            # Extract email address
            email_pattern = r'<(.+?)>'
            email_match = re.search(email_pattern, from_)
            sender_email = email_match.group(1) if email_match else from_
            
            # Get email body
            body = ""
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = email_message.get_payload(decode=True).decode()
            
            unread_emails.append({
                'subject': subject,
                'from': from_,
                'sender_email': sender_email,
                'sender_name': email_management.get_sender_name(from_),
                'body': body
            })
            
        imap.close()
        imap.logout()
        print(f"Fetched {len(unread_emails)} unread emails.")
        return unread_emails
        
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def send_email_response(email_info, subject, body):
    """Send email response to a specific recipient"""
    try:
        receiver = email_info["sender_email"]
        
        EMAIL_SENDER = 'mailreplyer5090@gmail.com'
        EMAIL_PASSWORD = 'wfkj ohob ezhi arux'
        em = EmailMessage()
        em['From'] = EMAIL_SENDER
        em['To'] = receiver
        em['Subject'] = subject
        em.set_content(body)
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, receiver, em.as_string())
            
        print(f"Email sent successfully to {receiver}")
        return True
        
    except Exception as e:
        print(f"Error sending email to {email_info['sender_email']}: {e}")
        return False

def process_single_email(email_info):
    """Process a single email using the existing graph"""
    try:
        # Create state for single email
        single_email_state = {
            "input_email": email_info["body"],
            "email_classifer": None,
            "next": None,
            "email_info": [email_info],
            "output_email_subject": None,
            "output_email_body": None
        }
        
        # Run classification
        classified_state = classifer(single_email_state)
        single_email_state.update(classified_state)
        
        # Route to appropriate handler
        routed_state = router(single_email_state)
        single_email_state.update(routed_state)
        
        # Process based on type
        if single_email_state["next"] == "type A":
            result = type_A_writer(single_email_state)
            single_email_state.update(result)
        elif single_email_state["next"] == "type B":
            result = type_B_writer(single_email_state)
            single_email_state.update(result)
        
        # Send the email
        success = send_email_response(
            email_info,
            single_email_state["output_email_subject"],
            single_email_state["output_email_body"]
        )
        
        return success
        
    except Exception as e:
        print(f"Error processing email from {email_info['sender_email']}: {e}")
        return False

def main_email_processing_loop():
    """Main loop that processes emails continuously"""
    print("Starting email processing system...")
    
    while True:
        try:
            # Fetch all unread emails
            email_queue = get_emails_batch()
            
            if not email_queue:
                print("No emails to process. Waiting 10 minutes...")
                time.sleep(600) 
                continue
            
            print(f"Processing {len(email_queue)} emails...")
            
            # Process each email one by one
            processed_count = 0
            failed_count = 0
            
            for i, email_info in enumerate(email_queue, 1):
                print(f"\n--- Processing email {i}/{len(email_queue)} ---")
                print(f"From: {email_info['sender_email']}")
                print(f"Subject: {email_info['subject']}")
                
                success = process_single_email(email_info)
                
                if success:
                    processed_count += 1
                    print(f"✓ Successfully processed and replied to email {i}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to process email {i}")
                
                # Small delay between emails to avoid overwhelming the system
                time.sleep(2)
            
            print(f"\n--- Batch Complete ---")
            print(f"Successfully processed: {processed_count}")
            print(f"Failed: {failed_count}")
            print(f"Total: {len(email_queue)}")
            
            # Wait 10 minutes before next batch
            print("Waiting 10 minutes before next email check...")
            time.sleep(600)
            
        except KeyboardInterrupt:
            print("\nEmail processing stopped by user.")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Waiting 10 minutes before retrying...")
            time.sleep(600)


graph = StateGraph(state)
graph.add_node("classifer", classifer)
graph.add_node("router", router)
graph.add_node("type_A_writer", type_A_writer)
graph.add_node("type_B_writer", type_B_writer)

graph.add_edge(START, "classifer")
graph.add_edge("classifer", "router")
graph.add_conditional_edges("router",
                            lambda state: state.get("next"),
                            {"type A": "type_A_writer",
                             "type B": "type_B_writer"
                             })
graph.add_edge("type_A_writer", END)
graph.add_edge("type_B_writer", END)

graph_compiled = graph.compile()


if __name__ == "__main__":
    main_email_processing_loop()