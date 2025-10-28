<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Infrabel_logo.svg" alt="Infrabel Logo" width="250"/>
</p>

<h1 align="center">🚆 Infrabot</h1>
<p align="center">Natural Language Interface for Railway Data Exploration</p>

**Infrabot** is a proof-of-concept (PoC) chatbot developed for **Infrabel**, the company responsible for managing Belgium’s railway network.  

Infrabel maintains extensive datasets containing detailed information about all trains (both Belgian and foreign) passing through its infrastructure. As of **May 2025**, retrieving specific information or statistics from these datasets required data scientists to manually write complex **SAS** queries.  

For example, answering a question such as:  
> *"How many freight trains passed through Brussels-South on weekends in June and July 2021?"*  

…would require a data scientist to craft and execute a custom **PROC SQL** query in SAS.

---

## 💡 Purpose

**Infrabot** simplifies this process by enabling data scientists to query railway data using **natural language** — in **English**, **French**, or **Dutch** — without writing any code.

---

## ⚙️ How It Works

1. The user types a question in natural language.  
2. Infrabot automatically:
   - Interprets the question and clarifies intent if needed,  
   - Generates an appropriate **SAS PROC SQL** query,  
   - Connects securely to Infrabel’s **SAS databases**,  
   - Retrieves and displays the results in a user-friendly format.  
3. The chatbot can also display geographic or route-based data on maps when relevant.

---

## 🧩 PoC Versions

Three different versions of **Infrabot** were developed to explore various interaction and reasoning approaches:

### **1️⃣ Single-Iteration Version**
A straightforward implementation that processes the user’s request in a single flow:  
**Question → (Optional) Clarification → PROC SQL generation → Data retrieval and display.**

---

### **2️⃣ Agentic (Class-Based) Version**
An agentic approach implemented through defined classes.  
This version allows the chatbot to **refine one user question into multiple PROC SQL generations**, enabling more accurate and complex data exploration.

---

### **3️⃣ Agentic Version with Google ADK**
The most advanced version, also agentic, integrates **Google’s state-of-the-art Agent Development Kit (ADK)**.  
This implementation leverages ADK’s orchestration capabilities for reasoning, task decomposition, and tool management — improving accuracy, flexibility, and scalability.

---

## 🚀 Key Features

- 🗣️ Multilingual support (English / French / Dutch)  
- 🤖 Automatic SAS PROC SQL generation  
- 🔗 Direct SAS database integration  
- 📊 Real-time data retrieval and visualization  
- 🧠 Agentic reasoning and multi-step refinement  
- 🗺️ Optional map-based display of results  
- 📧 Automatic email summary generation from query results  
- 📄 PDF export of results and summaries  

---

## 🧱 Tech Stack

**Frontend & Interface**
- **Streamlit** – interactive web UI for chat, controls, and data visualization  
- **Altair** – for dynamic bar, histogram, and time-series charts  
- **PyDeck** – for map visualizations of station and route data  

**Backend & Data Layer**
- **SASPy** – secure connection and query execution on Infrabel’s SAS Viya platform  
- **Pandas** – data processing and result handling  
- **ReportLab** – PDF report generation from query outputs  

**AI & Natural Language Understanding**
- **Google Gemini 2.0 Flash API** – natural language to SAS PROC SQL generation  
- **Google ADK (Agent Development Kit)** – agentic reasoning and task orchestration (for version 3)

---

## 📍 Status

This repository presents a **Proof of Concept (PoC)** demonstrating the feasibility of **natural-language-driven data interaction** with SAS-based railway systems.  
Future work includes enhancing agentic reasoning, optimizing query generation accuracy, and scaling the chatbot to production-level deployment within Infrabel’s data environment.

---

## 🖼️ Example Use

> **User:** How many freight trains passed through Brussels-South in June 2023?  
>  
> **Infrabot:**
> - Checks whether the question has all needed information to write a query and asks for clarification if needed.
> - Generates the corresponding **PROC SQL** query.  
> - Executes it on Infrabel’s SAS Viya database.  
> - Returns a table of results, interactive charts, and optional map visualizations.  
> - Offers an auto-generated email summary and downloadable PDF.

---

## 🧑‍💻 Authors

Developed by Paweł Sulewski.  
The project demonstrates how natural language and AI can simplify access to complex railway data systems.

---
