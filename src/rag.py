import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class RAG:
  def __init__(self, model_name):
    self.model = OllamaLLM(
      model = model_name,
      temperature = 0
    )
    self.template = """
    Anda adalah asisten AI akademik yang cerdas, adaptif, dan akurat untuk UIN Syarif Hidayatullah Jakarta. Tugas Anda adalah menjawab pertanyaan pengguna hanya berdasarkan dokumen referensi yang disediakan di bawah ini.

    [DOKUMEN REFERENSI]
    {context}

    [PERTANYAAN PENGGUNA]
    {question}

    [INSTRUKSI PENGERJAAN]
    1. Jawablah pertanyaan di atas dengan ringkas, jelas, dan akurat sesuai dengan fakta yang ada di dalam [DOKUMEN REFERENSI].
    2. Jangan mengada-ada atau memberikan informasi di luar konteks yang diberikan.
    3. Jika jawaban tidak dapat ditemukan di dalam dokumen referensi, katakan dengan sopan: "Maaf, saya tidak dapat menemukan informasi tersebut dalam Pedoman Akademik."
    4. Jangan menyebutkan istilah seperti "berdasarkan teks di atas", "parent chunk", atau "konteks yang diberikan" di dalam jawaban akhir Anda agar terlihat natural bagi pengguna.

    Jawaban:

    """

    self.prompt = ChatPromptTemplate.from_template(self.template)
    self.chain = self.prompt | self.model

  def get_answer(self, question, context):
    t0 = time.perf_counter()

    answer = self.chain.invoke({
      "context": context,
      "question": question
    })

    total_time = round(time.perf_counter() - t0, 2)

    return answer, total_time
  
rag = RAG("llama3.2") 