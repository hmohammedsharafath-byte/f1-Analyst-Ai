import pandas as pd
from vector1 import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# =============================
# LOAD DATASET (GROUND TRUTH)
# =============================
CSV_FILE = r"D:\Formula1_2025Season_RaceResults.csv"
df = pd.read_csv(CSV_FILE)

# =============================
# BUILD SAME PROMPT AS APP
# =============================
model = OllamaLLM(model="gemma3:latest")

template = """
You are a factual assistant for Formula 1 data.
Use ONLY the provided records.
If answer not found, say: NOT_FOUND

Records:
{records}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# =============================
# GENERATE TEST QUESTIONS
# =============================
questions = []
answers = []

# Take first 10 rows for testing
sample = df.head(10)

for _, row in sample.iterrows():
    q = f"What race was held on {row['date']}?"
    a = row["name"]
    questions.append(q)
    answers.append(a)

# =============================
# RUN TEST
# =============================
correct = 0

for i, q in enumerate(questions):
    print(f"\nQ{i+1}: {q}")
    
    records = retriever.invoke(q)
    response = chain.invoke({
        "records": records,
        "question": q
    }).strip()

    print("Expected:", answers[i])
    print("Bot:", response)

    if answers[i].lower() in response.lower():
        correct += 1
        print("✅ Correct")
    else:
        print("❌ Wrong")

# =============================
# FINAL SCORE
# =============================
total = len(questions)
accuracy = (correct / total) * 100

print("\n======================")
print(f"Accuracy: {accuracy:.2f}%")
print("======================")
