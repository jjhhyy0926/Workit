from __future__ import annotations

from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
import re


# ============================================================
# 1. 공통 결과 구조
# ============================================================

@dataclass
class AIReviewIssue:
    issue_type: str
    message: str
    section_title: str = ""
    sample: str = ""


@dataclass
class AIReviewReport:
    passed: bool
    target_type: str
    content_similarity: float
    expected_section_count: int
    parsed_section_count: int
    content_issues: List[AIReviewIssue] = field(default_factory=list)
    section_issues: List[AIReviewIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 2. 공통 파싱 품질 검수 엔진
# ============================================================

class BaseParsingAIReviewAgent:
    """
    공통 파싱 품질 검수 엔진.

    공통 검수사항:
    1. 원본과 파싱 결과의 내용 차이 검수
    2. 소제목별 분할 품질 검수
    """

    target_type = "base"

    def __init__(
        self,
        expected_titles: Optional[List[str]] = None,
        content_similarity_threshold: float = 0.97,
        block_similarity_threshold: float = 0.90,
    ):
        self.expected_titles = expected_titles
        self.content_similarity_threshold = content_similarity_threshold
        self.block_similarity_threshold = block_similarity_threshold

    def review(
        self,
        original_text: str,
        parsed_sections: Any,
    ) -> AIReviewReport:
        sections = self._normalize_parsed_sections(parsed_sections)
        parsed_text = self._join_sections(sections)

        content_similarity = self._similarity(original_text, parsed_text)

        content_issues = self._review_content_difference(
            original_text=original_text,
            parsed_text=parsed_text,
            content_similarity=content_similarity,
        )

        section_issues = self._review_section_split(
            original_text=original_text,
            sections=sections,
        )

        expected_titles = self._get_expected_titles(original_text)

        return AIReviewReport(
            passed=not content_issues and not section_issues,
            target_type=self.target_type,
            content_similarity=round(content_similarity, 4),
            expected_section_count=len(expected_titles),
            parsed_section_count=len(sections),
            content_issues=content_issues,
            section_issues=section_issues,
        )

    # ========================================================
    # Part 1. 원본 대비 내용 차이 검수
    # ========================================================

    def _review_content_difference(
        self,
        original_text: str,
        parsed_text: str,
        content_similarity: float,
    ) -> List[AIReviewIssue]:
        issues: List[AIReviewIssue] = []

        if not parsed_text.strip():
            issues.append(AIReviewIssue(
                issue_type="empty_parsed_text",
                message="파싱 결과 텍스트가 비어 있습니다.",
            ))
            return issues

        if content_similarity < self.content_similarity_threshold:
            issues.append(AIReviewIssue(
                issue_type="content_similarity_low",
                message=f"원본과 파싱 결과의 전체 유사도가 낮습니다. similarity={content_similarity:.4f}",
            ))

        issues.extend(self._find_missing_original_content(original_text, parsed_text))
        issues.extend(self._find_added_parsed_content(original_text, parsed_text))

        token_issue = self._find_important_token_difference(original_text, parsed_text)
        if token_issue:
            issues.append(token_issue)

        return issues

    def _find_missing_original_content(
        self,
        original_text: str,
        parsed_text: str,
        max_samples: int = 5,
    ) -> List[AIReviewIssue]:
        issues = []
        parsed_blocks = self._split_blocks(parsed_text)

        for original_block in self._split_blocks(original_text):
            if len(self._normalize_compare_text(original_block)) < 30:
                continue

            best_similarity = self._best_similarity(original_block, parsed_blocks)

            if best_similarity < self.block_similarity_threshold:
                issues.append(AIReviewIssue(
                    issue_type="missing_original_content",
                    message="원본에는 있으나 파싱 결과에서 누락되었거나 크게 달라진 내용이 있습니다.",
                    sample=original_block[:250],
                ))

            if len(issues) >= max_samples:
                break

        return issues

    def _find_added_parsed_content(
        self,
        original_text: str,
        parsed_text: str,
        max_samples: int = 5,
    ) -> List[AIReviewIssue]:
        issues = []
        original_blocks = self._split_blocks(original_text)

        for parsed_block in self._split_blocks(parsed_text):
            if len(self._normalize_compare_text(parsed_block)) < 30:
                continue

            best_similarity = self._best_similarity(parsed_block, original_blocks)

            if best_similarity < self.block_similarity_threshold:
                issues.append(AIReviewIssue(
                    issue_type="added_parsed_content",
                    message="파싱 결과에 원본에서 찾기 어려운 내용이 추가되어 있습니다.",
                    sample=parsed_block[:250],
                ))

            if len(issues) >= max_samples:
                break

        return issues

    def _find_important_token_difference(
        self,
        original_text: str,
        parsed_text: str,
    ) -> Optional[AIReviewIssue]:
        original_tokens = self._extract_important_tokens(original_text)
        parsed_tokens = self._extract_important_tokens(parsed_text)

        if original_tokens != parsed_tokens:
            return AIReviewIssue(
                issue_type="important_token_changed",
                message="원본과 파싱 결과의 숫자/날짜/금액/조문번호 등 중요 정보가 다릅니다.",
                sample=f"원본 토큰 일부: {original_tokens[:30]} / 파싱 토큰 일부: {parsed_tokens[:30]}",
            )

        return None

    # ========================================================
    # Part 2. 소제목별 분할 품질 검수
    # ========================================================

    def _review_section_split(
        self,
        original_text: str,
        sections: List[Dict[str, str]],
    ) -> List[AIReviewIssue]:
        issues: List[AIReviewIssue] = []

        if not sections:
            issues.append(AIReviewIssue(
                issue_type="empty_sections",
                message="소제목별 파싱 결과가 비어 있습니다.",
            ))
            return issues

        expected_titles = self._get_expected_titles(original_text)

        expected_norm = [
            self._normalize_title(title)
            for title in expected_titles
        ]

        parsed_norm = [
            self._normalize_title(section.get("title", ""))
            for section in sections
        ]

        for raw_title, norm_title in zip(expected_titles, expected_norm):
            if norm_title and norm_title not in parsed_norm:
                issues.append(AIReviewIssue(
                    issue_type="missing_section",
                    section_title=raw_title,
                    message=f"원본의 소제목 '{raw_title}'이 파싱 결과에 없습니다.",
                ))

        issues.extend(self._check_section_order(expected_norm, parsed_norm))
        issues.extend(self._check_empty_section_content(sections))
        issues.extend(self._check_overmerged_sections(sections))

        return issues

    def _check_section_order(
        self,
        expected_titles_norm: List[str],
        parsed_titles_norm: List[str],
    ) -> List[AIReviewIssue]:
        parsed_existing = [
            title for title in parsed_titles_norm
            if title in expected_titles_norm
        ]

        expected_existing = [
            title for title in expected_titles_norm
            if title in parsed_existing
        ]

        if parsed_existing and parsed_existing != expected_existing:
            return [AIReviewIssue(
                issue_type="section_order_mismatch",
                message="소제목 순서가 원본과 다릅니다.",
                sample=f"expected={expected_existing[:20]}, parsed={parsed_existing[:20]}",
            )]

        return []

    def _check_empty_section_content(
        self,
        sections: List[Dict[str, str]],
    ) -> List[AIReviewIssue]:
        issues = []

        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")

            if title and not content.strip():
                issues.append(AIReviewIssue(
                    issue_type="empty_section_content",
                    section_title=title,
                    message=f"소제목 '{title}'은 존재하지만 내용이 비어 있습니다.",
                ))

        return issues

    def _check_overmerged_sections(
        self,
        sections: List[Dict[str, str]],
    ) -> List[AIReviewIssue]:
        issues = []

        section_titles = [
            section.get("title", "")
            for section in sections
            if section.get("title")
        ]

        for section in sections:
            current_title = section.get("title", "")
            current_norm = self._normalize_title(current_title)
            content_norm = self._normalize_compare_text(section.get("content", ""))

            for other_title in section_titles:
                other_norm = self._normalize_title(other_title)

                if not other_norm or other_norm == current_norm:
                    continue

                if other_norm in content_norm:
                    issues.append(AIReviewIssue(
                        issue_type="overmerged_section",
                        section_title=current_title,
                        message=(
                            f"소제목 '{current_title}' 내용 안에 "
                            f"다른 소제목 '{other_title}'이 섞여 있습니다."
                        ),
                    ))
                    break

        return issues

    # ========================================================
    # 입력 정규화
    # ========================================================

    def _normalize_parsed_sections(self, parsed_sections: Any) -> List[Dict[str, str]]:
        if not parsed_sections:
            return []

        if isinstance(parsed_sections, str):
            return [{
                "title": "",
                "content": parsed_sections.strip(),
            }]

        if isinstance(parsed_sections, dict):
            parsed_sections = [parsed_sections]

        sections = []

        for idx, item in enumerate(parsed_sections, start=1):
            if isinstance(item, str):
                sections.append({
                    "title": f"섹션{idx}",
                    "content": item.strip(),
                })
                continue

            if not isinstance(item, dict):
                continue

            title = (
                item.get("title")
                or item.get("heading")
                or self._build_title_from_item(item)
                or item.get("section")
                or item.get("chunk_id")
                or f"섹션{idx}"
            )

            content = (
                item.get("content")
                or item.get("text")
                or item.get("chunk_text")
                or item.get("body")
                or item.get("section_text")
                or ""
            )

            sections.append({
                "title": str(title).strip(),
                "content": str(content).strip(),
            })

        return sections

    def _build_title_from_item(self, item: Dict[str, Any]) -> str:
        return ""

    def _join_sections(self, sections: List[Dict[str, str]]) -> str:
        lines = []

        for section in sections:
            title = section.get("title", "").strip()
            content = section.get("content", "").strip()

            if title and content:
                lines.append(f"{title}\n{content}")
            elif title:
                lines.append(title)
            elif content:
                lines.append(content)

        return "\n".join(lines)

    # ========================================================
    # 확장 포인트
    # ========================================================

    def _get_expected_titles(self, original_text: str) -> List[str]:
        if self.expected_titles is not None:
            return self.expected_titles

        return []

    def _extract_important_tokens(self, text: str) -> List[str]:
        patterns = [
            r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일",
            r"\d+\s*일",
            r"\d+\s*개월",
            r"\d+\s*년",
            r"\d+\s*원",
            r"\d+\s*만원",
            r"\d+\s*억원",
            r"\d+(?:\.\d+)?\s*%",
            r"\d+",
        ]

        return self._find_tokens(text, patterns)

    # ========================================================
    # 공통 유틸
    # ========================================================

    def _split_blocks(self, text: str) -> List[str]:
        text = self._clean_text(text)

        blocks = re.split(r"\n\s*\n", text)

        if len(blocks) <= 1:
            blocks = re.split(r"(?<=[.!?。！？]|다\.|함\.|음\.)\s+", text)

        return [block.strip() for block in blocks if block.strip()]

    def _best_similarity(self, target: str, candidates: List[str]) -> float:
        target_norm = self._normalize_compare_text(target)
        best = 0.0

        for candidate in candidates:
            candidate_norm = self._normalize_compare_text(candidate)
            score = SequenceMatcher(None, target_norm, candidate_norm).ratio()
            best = max(best, score)

        return best

    def _find_tokens(self, text: str, patterns: List[str]) -> List[str]:
        tokens = []

        for pattern in patterns:
            tokens.extend(re.findall(pattern, text or ""))

        return [
            self._normalize_compare_text(token)
            for token in tokens
            if self._normalize_compare_text(token)
        ]

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(
            None,
            self._normalize_compare_text(a),
            self._normalize_compare_text(b),
        ).ratio()

    def _normalize_title(self, title: str) -> str:
        title = title or ""
        title = re.sub(r"\s+", "", title)
        title = re.sub(r"[()（）\[\]【】ㆍ·.,:;]", "", title)
        return title.strip()

    def _normalize_compare_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[()（）\[\]【】ㆍ·.,:;]", "", text)
        return text.strip()

    def _clean_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ============================================================
# 3. 법령용 에이전트
# ============================================================

class LawParsingAIReviewAgent(BaseParsingAIReviewAgent):
    """
    법령 파싱 품질 검수 에이전트.

    Workit 법령 RAG용:
    - 제1장
    - 제1절
    - 제1조(목적)
    - 제2조(정의)
    같은 법령 소제목 구조를 검수합니다.
    """

    target_type = "law"

    def _get_expected_titles(self, original_text: str) -> List[str]:
        if self.expected_titles is not None:
            return self.expected_titles

        patterns = [
            r"제\s*\d+\s*장\s*[^\n]+",
            r"제\s*\d+\s*절\s*[^\n]+",
            r"제\s*\d+\s*조(?:\s*의\s*\d+)?\s*\([^)]*\)",
        ]

        found = []

        for pattern in patterns:
            for match in re.finditer(pattern, original_text or ""):
                title = re.sub(r"\s+", " ", match.group(0)).strip()
                found.append((match.start(), title))

        found.sort(key=lambda item: item[0])

        result = []
        seen = set()

        for _, title in found:
            norm = self._normalize_title(title)
            if norm not in seen:
                result.append(title)
                seen.add(norm)

        return result

    def _build_title_from_item(self, item: Dict[str, Any]) -> str:
        article = item.get("article") or item.get("article_number") or item.get("조") or ""
        article_title = item.get("article_title") or item.get("title") or ""
        paragraph = item.get("paragraph") or item.get("항") or ""
        subparagraph = item.get("subparagraph") or item.get("호") or ""

        title = str(article).strip()

        if article_title:
            title += f"({article_title})"

        if paragraph:
            title += str(paragraph).strip()

        if subparagraph:
            title += str(subparagraph).strip()

        return title.strip()

    def _extract_important_tokens(self, text: str) -> List[str]:
        patterns = [
            r"제\s*\d+\s*장",
            r"제\s*\d+\s*절",
            r"제\s*\d+\s*조(?:\s*의\s*\d+)?",
            r"제\s*\d+\s*항",
            r"제\s*\d+\s*호",
            r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일",
            r"\d+\s*일",
            r"\d+\s*개월",
            r"\d+\s*년",
            r"\d+\s*원",
            r"\d+\s*만원",
            r"\d+\s*억원",
            r"\d+(?:\.\d+)?\s*%",
            r"\d+",
        ]

        return self._find_tokens(text, patterns)


# ============================================================
# 4. 산출물용 에이전트
# ============================================================

DEFAULT_DELIVERABLE_TITLES = [
    "사업명",
    "사업기간",
    "사업목적",
    "사업범위",
    "사업추진체계",
    "사업추진절차",
    "산출물계획",
    "일정계획",
    "공정별 투입인력계획",
    "보고계획",
    "표준화계획",
    "품질보증계획",
    "위험관리계획",
    "보안대책",
    "교육계획",
    "발주기관 협조요청사항",
]


class DeliverableParsingAIReviewAgent(BaseParsingAIReviewAgent):
    """
    산출물 파싱 품질 검수 에이전트.

    Workit 산출물 품질평가용:
    - 사업명
    - 사업기간
    - 사업목적
    - 사업범위
    같은 산출물 소제목 구조를 검수합니다.
    """

    target_type = "deliverable"

    def __init__(
        self,
        expected_titles: Optional[List[str]] = None,
        content_similarity_threshold: float = 0.95,
        block_similarity_threshold: float = 0.90,
    ):
        super().__init__(
            expected_titles=expected_titles or DEFAULT_DELIVERABLE_TITLES,
            content_similarity_threshold=content_similarity_threshold,
            block_similarity_threshold=block_similarity_threshold,
        )

    def _build_title_from_item(self, item: Dict[str, Any]) -> str:
        return str(
            item.get("title")
            or item.get("section_title")
            or item.get("heading")
            or item.get("name")
            or item.get("section")
            or ""
        ).strip()

    def _extract_important_tokens(self, text: str) -> List[str]:
        patterns = [
            r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일",
            r"\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}",
            r"\d+\s*일",
            r"\d+\s*개월",
            r"\d+\s*년",
            r"\d+\s*원",
            r"\d+\s*만원",
            r"\d+\s*억원",
            r"\d+(?:\.\d+)?\s*%",
            r"\d+",
        ]

        return self._find_tokens(text, patterns)


# ============================================================
# 5. 호출용 함수
# ============================================================

def review_law_parsing(
    original_text: str,
    parsed_sections: Any,
) -> Dict[str, Any]:
    agent = LawParsingAIReviewAgent()
    return agent.review(
        original_text=original_text,
        parsed_sections=parsed_sections,
    ).to_dict()


def review_deliverable_parsing(
    original_text: str,
    parsed_sections: Any,
    expected_titles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    agent = DeliverableParsingAIReviewAgent(
        expected_titles=expected_titles,
    )
    return agent.review(
        original_text=original_text,
        parsed_sections=parsed_sections,
    ).to_dict()


def review_workit_parsing(
    target_type: str,
    original_text: str,
    parsed_sections: Any,
    expected_titles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    통합 호출 함수.

    target_type:
    - "law"
    - "deliverable"
    """

    if target_type == "law":
        return review_law_parsing(
            original_text=original_text,
            parsed_sections=parsed_sections,
        )

    if target_type == "deliverable":
        return review_deliverable_parsing(
            original_text=original_text,
            parsed_sections=parsed_sections,
            expected_titles=expected_titles,
        )

    raise ValueError(f"지원하지 않는 target_type입니다: {target_type}")