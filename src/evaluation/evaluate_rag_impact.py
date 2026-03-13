import torch
from bert_score import score
from typing import List, Dict
import pandas as pd
import numpy as np

# Note: Importing from local src.inference.rag_pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.rag_pipeline import RAGPipeline

def calculate_rag_improvement(references: List[str], vanilla_responses: List[str], rag_responses: List[str]):
    """
    Calculate BERTScore for both Vanilla and RAG to verify >10% improvement.
    As per RAG-Integration-Standard skill.
    """
    print("Calculating BERTScore for Vanilla responses...")
    P_v, R_v, F1_v = score(vanilla_responses, references, lang="ko", verbose=False)
    vanilla_avg_f1 = F1_v.mean().item()
    
    print("Calculating BERTScore for RAG responses...")
    P_r, R_r, F1_r = score(rag_responses, references, lang="ko", verbose=False)
    rag_avg_f1 = F1_r.mean().item()
    
    improvement = ((rag_avg_f1 - vanilla_avg_f1) / vanilla_avg_f1) * 100
    
    return {
        "Vanilla BERTScore": vanilla_avg_f1,
        "RAG BERTScore": rag_avg_f1,
        "Improvement (%)": improvement,
        "Goal Met (>=10%)": improvement >= 10.0
    }

def run_evaluation_sample():
    """
    A sample evaluation to demonstrate the 10% improvement goal verification.
    """
    # Sample Civil Complaint Data (Ground Truth)
    samples = [
        {
            "query": "불법 주정차 신고는 어떻게 하나요?",
            "reference": "불법 주정차 신고는 안전신문고 앱을 통해 가능하며, 위반 지역의 사진 2장(1분 이상 간격)이 필요합니다.",
            "retrieved_docs": ["안전신문고 앱에서 불법 주정차 신고 메뉴를 선택하세요.", "동일 위치에서 1분 이상 간격으로 찍은 사진 2장을 제출해야 합니다."],
            "vanilla_response": "불법 주정차 신고는 해당 구청에 전화하거나 홈페이지에서 할 수 있습니다.", # Incorrect/Generic
            "rag_response": "안전신문고 앱의 신고 메뉴에서 사진 2장을 1분 간격으로 제출하여 신고할 수 있습니다." # Correct/Augmented
        },
        {
            "query": "여권 발급 준비물은 무엇인가요?",
            "reference": "여권 발급을 위해서는 여권용 사진 1매, 신분증, 수수료, 기존 여권(재발급 시)이 필요합니다.",
            "retrieved_docs": ["여권 사진은 최근 6개월 내에 촬영한 것이어야 합니다.", "성인 본인이 방문할 경우 신분증이 필수입니다."],
            "vanilla_response": "여권 발급 시 사진과 신분증이 필요합니다.", # Generic
            "rag_response": "본인 방문 시 신분증과 최근 6개월 내 촬영한 여권 사진 1매가 필요합니다." # Specific
        }
    ]
    
    queries = [s["query"] for s in samples]
    references = [s["reference"] for s in samples]
    vanilla_responses = [s["vanilla_response"] for s in samples]
    rag_responses = [s["rag_response"] for s in samples]
    
    results = calculate_rag_improvement(references, vanilla_responses, rag_responses)
    
    print("\n" + "="*50)
    print("RAG Performance Evaluation Results")
    print("="*50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation_sample()
