import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from typing import List, Dict

class RAGPipeline:
    def __init__(self, model_id: str = "LG-EXAONE/EXAONE-3.0-7.8B-Instruct", 
                 embedding_model_id: str = "intfloat/multilingual-e5-large",
                 faiss_index_path: str = None):
        """
        RAG Pipeline for Civil Complaint System.
        - EXAONE 32K context window
        - Multilingual-E5 embedding
        - FAISS retrieval
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
        self.embed_model = AutoModel.from_pretrained(embedding_model_id)
        
        # Load FAISS index if provided
        if faiss_index_path:
            self.index = faiss.read_index(faiss_index_path)
        else:
            self.index = None
            
        self.context_window = 32768  # EXAONE 32K

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using multilingual-e5-large."""
        inputs = self.embed_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def search(self, query: str, k: int = 5) -> List[str]:
        """Search similar documents in FAISS index."""
        if not self.index:
            return ["(참고문서 데이터가 아직 로드되지 않았습니다.)"]
        
        query_vector = self.generate_embeddings([f"query: {query}"])
        distances, indices = self.index.search(query_vector, k)
        
        # Note: In a real implementation, indices should be mapped to actual text documents.
        # This is a placeholder for integration.
        return [f"유사 문서 {i}" for i in indices[0]]

    def augment_prompt(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Augment prompt with retrieved documents for EXAONE 32K.
        Follows the RAG-Integration-Standard skill pattern.
        """
        context = "\n".join([f"참고문서 {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""당신은 민원 처리 전문가입니다. 아래 제공된 [참고문서]의 내용을 바탕으로 사용자의 [질문]에 답변하세요.
반드시 [참고문서]에 근거하여 답변하고, 문서에 없는 내용은 답변하지 마세요.

[참고문서]
{context}

[질문]
{query}

[답변]
"""
        return prompt

    def generate_response(self, query: str, retrieved_docs: List[str]):
        """Placeholder for actual LLM generation (vLLM or HF)."""
        prompt = self.augment_prompt(query, retrieved_docs)
        # This will be connected to vLLM or Hugging Face model inference
        return prompt # Returning prompt for now to verify augmentation

if __name__ == "__main__":
    # Simple test for RAG pipeline logic
    rag = RAGPipeline()
    query = "불법 주정차 신고는 어떻게 하나요?"
    docs = ["불법 주정차는 안전신문고 앱을 통해 신고 가능합니다.", "신고 시 사진 2장이 필요합니다."]
    augmented = rag.augment_prompt(query, docs)
    print("--- Augmented Prompt ---")
    print(augmented)
