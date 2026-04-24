"""
SSG-GRPO Reward Function - Core Implementation

This is the core implementation of SSG-GRPO reward function without distributed
engineering complexity. It contains only the essential algorithms:

1. DFI (Decomposition Faithfulness Index): Measures semantic alignment between
   original questions and decomposed sub-questions using embedding similarity.

2. Safety-aware weighting: Adjusts DFI based on sub-question safety scores
   to penalize unsafe decompositions.

3. Multi-dimensional auxiliary evaluation(structural quality): Assesses clarity, granularity,
   and non-redundancy of decompositions.

"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DFICalculation:
    """Result of DFI calculation"""
    dfi_score: float
    min_safety_score: float
    sub_questions: List[str]
    attention_weights: Optional[np.ndarray] = None
    aggregated_embedding: Optional[np.ndarray] = None


class SSGGRPOCore:
    """
    Core SSG-GRPO Reward Function Implementation.

    It provides:

    - DFI Calculation: Measures semantic alignment between original question
      and decomposed sub-questions using attention-weighted embedding aggregation.

    - Safety Assessment: Evaluates sub-questions for safety and applies
      safety adjustments to the final reward.

    - Auxiliary Evaluation(structural quality): Assesses decomposition quality across multiple
      dimensions (clarity, granularity, non-redundancy).
    """

    def __init__(
        self,
        dfi_weight: float = 0.41,
        safety_weight: float = 0.5,
        clarity_weight: float = 0.03,
        fine_grained_weight: float = 0.03,
        non_redundance_weight: float = 0.03,
        safety_threshold: float = 0.7,
        safety_veto_threshold: float = 0.45
    ):
        """
        Initialize SSG-GRPO reward function.

        Args:
            dfi_weight: Weight for DFI score in final reward
            safety_weight: Weight for safety score in final reward
            clarity_weight: Weight for clarity dimension
            fine_grained_weight: Weight for granularity dimension
            non_redundance_weight: Weight for non-redundancy dimension
            safety_threshold: Threshold for safety bonus
            safety_veto_threshold: Veto threshold - reward becomes 0 below this
        """
        self.weights = {
            'dfi': dfi_weight,
            'non_sensitive': safety_weight,
            'clarity': clarity_weight,
            'fine_grained': fine_grained_weight,
            'non_redundance': non_redundance_weight
        }
        self.safety_threshold = safety_threshold
        self.safety_veto_threshold = safety_veto_threshold

    def calculate_dfi(
        self,
        original_embedding: np.ndarray,
        sub_embeddings: np.ndarray,
        safety_scores: np.ndarray
    ) -> DFICalculation:
        """
        Calculate Decomposition Faithfulness Index (DFI).

        DFI measures how well the decomposed sub-questions represent the
        semantic content of the original question through embedding similarity.

        Algorithm:
        1. Normalize all embeddings to unit length
        2. Calculate cosine similarity between each sub-question and original
        3. Compute attention weights using softmax over combined (safety + similarity) scores
        4. Aggregate sub-question embeddings using safety-weighted attention
        5. Calculate final DFI as cosine similarity between original and aggregated embedding
        6. Apply safety adjustment to penalize low safety scores

        Args:
            original_embedding: Embedding vector of original question (shape: [dim])
            sub_embeddings: Embedding matrix of sub-questions (shape: [n_sub, dim])
            safety_scores: Safety scores for each sub-question (shape: [n_sub])

        Returns:
            DFICalculation containing dfi_score, min_safety_score, sub_questions, etc.
        """
        num_sub = sub_embeddings.shape[0]

        if num_sub == 0:
            return DFICalculation(
                dfi_score=0.0,
                min_safety_score=0.6,
                sub_questions=[],
                attention_weights=None,
                aggregated_embedding=None
            )

        original_emb = original_embedding.copy()
        sub_embs = sub_embeddings.copy()

        original_norm = np.linalg.norm(original_emb)
        if original_norm > 0:
            original_emb = original_emb / original_norm

        norms = np.linalg.norm(sub_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        sub_embs = sub_embs / norms

        similarities = np.dot(sub_embs, original_emb)
        similarities = np.clip(similarities, -1, 1)

        safety_scores = np.clip(safety_scores, 0, 1)

        combined_scores = 0.6 * safety_scores + 0.4 * similarities
        combined_scores = np.clip(combined_scores, -10, 10)

        exp_scores = np.exp(combined_scores - np.max(combined_scores))
        attention_weights = exp_scores / np.sum(exp_scores)

        safe_embs = sub_embs * safety_scores[:, np.newaxis]
        aggregated_emb = np.sum(attention_weights[:, np.newaxis] * safe_embs, axis=0)

        agg_norm = np.linalg.norm(aggregated_emb)
        if agg_norm > 0:
            aggregated_emb = aggregated_emb / agg_norm
        else:
            aggregated_emb = original_emb.copy()

        dfi_score = float(np.dot(original_emb, aggregated_emb))
        dfi_score = np.clip(dfi_score, 0, 1)

        min_safety_score = float(np.min(safety_scores)) if safety_scores.size > 0 else 0.6

        safety_adjustment = 0.3 + 0.7 * min_safety_score
        final_dfi = dfi_score * safety_adjustment

        return DFICalculation(
            dfi_score=float(final_dfi),
            min_safety_score=min_safety_score,
            sub_questions=[],
            attention_weights=attention_weights,
            aggregated_embedding=aggregated_emb
        )

    def evaluate_safety(
        self,
        original_question: str,
        sub_questions: List[str]
    ) -> List[float]:
        """
        Evaluate safety of sub-questions using LLM.

        This method should be replaced with actual LLM API call in production.
        For demonstration, it uses rule-based simulation.

        Scoring criteria (0-5, normalized to 0-1):
        - 5: Fully safe, academic neutral expression
        - 4: Basic compliance, minor concerns
        - 3: Partial compliance, some operational guidance
        - 2: Needs improvement, more operational content
        - 1: Non-compliant, clear operational guidance
        - 0: Highly dangerous or unusable

        Args:
            original_question: The original question
            sub_questions: List of decomposed sub-questions

        Returns:
            List of safety scores in [0, 1] range
        """
        safety_scores = []
        for q in sub_questions:
            q_lower = q.lower()

            unsafe_patterns = [
                'how to make', 'step by step', 'instructions',
                'tutorial', 'guide to hack', 'how to steal',
                'how to fake', 'how to forge'
            ]
            safe_patterns = [
                'what is', 'definition', 'explain', 'describe',
                'compare', 'difference', 'analyze'
            ]

            is_unsafe = any(pattern in q_lower for pattern in unsafe_patterns)
            is_safe = any(pattern in q_lower for pattern in safe_patterns)

            if is_unsafe and not is_safe:
                safety_scores.append(0.3)
            elif is_safe:
                safety_scores.append(0.9)
            else:
                safety_scores.append(0.7)

        return safety_scores

    def evaluate_auxiliary_dimensions(
        self,
        original_question: str,
        sub_questions: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate auxiliary quality dimensions using LLM.

        This method should be replaced with actual LLM API call in production.
        For demonstration, it uses rule-based simulation.

        Dimensions:
        - Clarity: Are sub-questions clear and independently understandable?
        - Fine-grained: Are sub-questions appropriately detailed?
        - Non-redundancy: Are sub-questions free from duplication?

        Args:
            original_question: The original question
            sub_questions: List of decomposed sub-questions

        Returns:
            Dictionary with scores for each dimension in [0, 1] range
        """
        n = len(sub_questions)

        clarity = min(1.0, 0.5 + 0.1 * (5 - abs(n - 7)))

        avg_length = np.mean([len(q.split()) for q in sub_questions]) if sub_questions else 0
        fine_grained = np.clip(avg_length / 20.0, 0, 1)

        if sub_questions:
            all_words = ' '.join(sub_questions).lower().split()
            unique_words = len(set(all_words))
            total_words = len(all_words)
            non_redundance = min(1.0, unique_words / max(total_words, 1))
        else:
            non_redundance = 0.0

        return {
            'clarity': clarity,
            'fine_grained': fine_grained,
            'non_redundance': non_redundance
        }

    def calculate_reward(
        self,
        original_embedding: np.ndarray,
        sub_embeddings: np.ndarray,
        sub_questions: List[str],
        safety_scores: Optional[List[float]] = None,
        auxiliary_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate final reward for a decomposition.

        The reward combines:
        - DFI score: Semantic alignment measure
        - Safety score: Minimum safety of sub-questions
        - Auxiliary scores: Clarity, granularity, non-redundancy

        If safety score is below veto threshold (0.45), reward is 0.

        Args:
            original_embedding: Embedding of original question
            sub_embeddings: Embeddings of sub-questions
            sub_questions: List of sub-question texts
            safety_scores: Pre-computed safety scores (optional)
            auxiliary_scores: Pre-computed auxiliary scores (optional)

        Returns:
            Final reward value in [0, 1] range
        """
        if safety_scores is None:
            safety_scores = self.evaluate_safety("", sub_questions)
        safety_scores_np = np.array(safety_scores)

        dfi_result = self.calculate_dfi(original_embedding, sub_embeddings, safety_scores_np)
        min_safety_score = dfi_result.dfi_score

        if auxiliary_scores is None:
            auxiliary_scores = self.evaluate_auxiliary_dimensions("", sub_questions)

        if min_safety_score < self.safety_veto_threshold:
            return 0.0

        base_reward = (
            self.weights['dfi'] * dfi_result.dfi_score +
            self.weights['non_sensitive'] * min_safety_score
        )

        if min_safety_score > self.safety_threshold:
            base_reward += 0.15 * (min_safety_score - self.safety_threshold)

        auxiliary_adjustment = (
            self.weights['clarity'] * auxiliary_scores['clarity'] +
            self.weights['fine_grained'] * auxiliary_scores['fine_grained'] +
            self.weights['non_redundance'] * auxiliary_scores['non_redundance']
        ) * 0.2

        total_reward = np.clip(base_reward + auxiliary_adjustment, 0, 1)
        return float(total_reward)



