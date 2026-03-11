# Accelerating-Action-Robust-Deep-Deterministic-Policy-Gradient
Code for Accelerating Action-Robust Deep Deterministic Policy Gradient (AAMAS 2026, EA, To appear).

이 저장소는 **Walker2d-v4**를 중심으로 학습 및 평가 파이프라인을 제공합니다.  
핵심 아이디어는 논문과 같이 **별도의 adversarial policy network를 학습하지 않고**, 행동공간의 **vertex(꼭짓점)** 들을 평가하여 critic 값을 가장 작게 만드는 adversarial action을 선택하는 것입니다.

> **주의**
> - 이 코드는 **EAR-DDPG의 핵심 아이디어**를 반영하도록 정리한 구현입니다.
> - 현재 스크립트는 주로 **Walker2d-v4** 기준으로 구성되어 있습니다.
> - 학습, 체크포인트 저장, heatmap 기반 강건성 평가에는 사용할 수 있지만, 논문의 **모든 실험을 완전히 자동 재현하는 패키지**는 아닙니다.

---

## 1. 개요

Noisy-action robust RL에서는 실제로 환경에 입력되는 행동이 다음과 같이 정의됩니다.

\[
\tilde{a} = (1-\alpha)a_P + \alpha a_A
\]

여기서:

- \(a_P\): protagonist action
- \(a_A\): adversarial action
- \(\alpha\): mixing coefficient

이 구현의 핵심은 다음과 같습니다.

1. **protagonist actor**와 **critic**만 학습한다.
2. 별도의 adversary network를 학습하지 않는다.
3. adversarial action은 행동공간 hypercube의 **모든 vertex**를 평가해서 구한다.
4. 그중 **Q-value를 최소화하는 vertex**를 선택한다.

즉, 구조를 단순하게 유지하면서도 EAR-DDPG의 핵심 직관을 살리는 구현입니다.

---

## 2. 주요 특징

- **PyTorch 기반** EAR-DDPG 스타일 구현
- **Gym** 기반 학습 루프
- 기본 환경: **Walker2d-v4**
- **Actor / Critic 체크포인트 저장**
- **`[-max_action, max_action]^N`** 기준 adversarial candidate 생성

---

## 3. 저장소 구조

```text
.
├── FAR_DDPG.py         # agent, replay buffer, vertex search, training step
├── train_FAR_DDPG.py   # training script
├── eval_policy.py      # robustness heatmap evaluation script
├── utils.py            # actor, critic, evaluation, utility functions
└── README.md