# test_rag.py
import sys
sys.path.insert(0, 'rag')

from inference import load_model, predict

print("모델 로딩 중... (처음엔 수분 소요)")
model, tokenizer = load_model()
print("로딩 완료!")

# 아까 RAG에서 나온 law_refs 샘플 그대로 사용
law_refs = [
    {
        "source_full": "지방자치단체 용역계약 일반조건 제8절 제7항 가",
        "chunk_text": "계약상대자는 계약의 수행 중 기술용역 목적물 및 제3자에 대한 손해를 부담하여야 한다. 다만, 계약상대자의 책임없는 사유로 인하여 발생한 경우에는 발주기관의 부담으로 한다."
    }
]

result = predict(
    clause_text="제7조(손해배상) 을의 귀책사유로 인한 손해배상은 계약금액의 10%를 초과할 수 없다.",
    law_refs=law_refs,
    model=model,
    tokenizer=tokenizer,
)

print("\n=== sLLM 판정 결과 ===")
print(result)