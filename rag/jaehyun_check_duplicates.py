"""
Workit - chunk 중복 검사 스크립트
structured/ 폴더의 JSON을 읽어서:

  [핵심] chunk_id 중복 — 같은 chunk_id가 한 파일 안에 두 번 이상 존재.
         이건 무조건 파싱 버그(원본 docx 중복, 시행일 예고 병기 미처리 등).

  [참고] 텍스트만 동일, chunk_id는 다름 — 법 조문 안에서 "삭제", "가. 국가" 같은
         표준 문구/짧은 목록이 여러 조항에 자연스럽게 반복되는 정상 현상.
         에러 아님. 다만 같은 chunk_id 중복이 0건이 아니라면 먼저 그것부터 잡을 것.

사용법:
    python jaehyun_check_duplicates.py
"""

import sys
import io
import json
from collections import Counter
from pathlib import Path

# Windows 콘솔(cp949) 에서도 한자ㆍ특수문자 깨지지 않게 출력 인코딩 강제
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STRUCTURED_DIR = Path("C:/lecture/Workit/data/structured")


def load_all_articles() -> list[dict]:
    all_articles = []
    for path in sorted(STRUCTURED_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for a in data.get("articles", []):
            all_articles.append({
                "file":     path.name,
                "chunk_id": a.get("chunk_id"),
                "law_name": a.get("law_name"),
                "text":     (a.get("text") or "").strip(),
            })
    return all_articles


def check_chunk_id_duplicates(articles: list[dict]) -> int:
    """[핵심] 같은 파일 안에서 chunk_id가 중복되는 경우 — 실제 버그"""
    by_file: dict[str, list[dict]] = {}
    for a in articles:
        by_file.setdefault(a["file"], []).append(a)

    total = 0
    print("\n" + "=" * 60)
    print("[핵심 검사] chunk_id 중복 (파일 내 동일 chunk_id 2회 이상)")
    print("=" * 60)

    for filename, items in by_file.items():
        ids = [a["chunk_id"] for a in items]
        c = Counter(ids)
        dups = {k: v for k, v in c.items() if v > 1}
        if not dups:
            continue
        print(f"\n[{filename}] {len(dups)}개 chunk_id 중복")
        for chunk_id, count in dups.items():
            print(f"    {chunk_id} × {count}")
            total += 1

    if total == 0:
        print("\n  [OK] chunk_id 중복 없음 - 정상")
    else:
        print(f"\n  [WARN] 총 {total}개 chunk_id 중복 발견 - 파싱 로직 확인 필요")

    return total


def check_same_text_diff_id(articles: list[dict]) -> int:
    """[참고] 텍스트는 같지만 chunk_id가 다른 경우 — 대부분 정상(법 조문 반복 문구)"""
    text_to_chunks: dict[str, list[dict]] = {}
    for a in articles:
        if not a["text"]:
            continue
        text_to_chunks.setdefault(a["text"], []).append(a)

    dups = {t: lst for t, lst in text_to_chunks.items() if len(lst) > 1}

    print("\n" + "=" * 60)
    print(f"[참고] 텍스트 동일·chunk_id 다름: {len(dups)}건")
    print("  → 법 조문 안에서 반복되는 표준 문구(예: '③ 삭제', '가. 국가')일 가능성이 높음.")
    print("    버그 아님. chunk_id 중복이 0건이면 신경 쓰지 않아도 됨.")
    print("=" * 60)
    for text, lst in dups.items():
        ids = [f'{a["chunk_id"]}({a["file"]})' for a in lst]
        preview = text[:50].replace("\n", " ")
        print(f"  [{len(lst)}개] {preview}... → {ids}")

    return len(dups)


def main():
    articles = load_all_articles()
    print(f"총 {len(articles)}개 chunk 로드 완료")

    dup_id_count = check_chunk_id_duplicates(articles)
    same_text_count = check_same_text_diff_id(articles)

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"  chunk_id 중복(버그):        {dup_id_count}건")
    print(f"  텍스트만 동일(정상/참고용): {same_text_count}건")
    print("\n검사 완료")


if __name__ == "__main__":
    main()
