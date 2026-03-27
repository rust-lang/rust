---
name: bci-brain-state
description: Interprets real-time brain-computer interface (BCI) data injected into context. Helps adapt responses based on the user's cognitive state.
version: 0.1.0
user-invocable: false
disable-model-invocation: false
---

# Brain-Computer Interface (BCI) State Interpretation

## What is BCI data?

An EEG headset measures the user's brain electrical activity in real time. Raw signals are processed into frequency bands (delta, theta, alpha, beta, gamma), which are then used to classify the user's mental state and derive cognitive scores. This data is injected into your context as a natural language summary before each response.

## Brain states

The `state.primary` field contains the classified mental state. Adapt your responses accordingly:

- **focused** -- High beta/theta ratio. The user is concentrated and engaged. Provide detailed, thorough responses. They can handle complexity.
- **relaxed** -- High alpha power. The user is calm. A conversational, easy-going tone works well.
- **stressed** -- High beta with low alpha. The user may be frustrated or under pressure. Be concise, supportive, and clear. Avoid overwhelming them with information.
- **drowsy** -- High theta power. The user is fatigued. Keep responses short and to the point. If appropriate, gently suggest taking a break.
- **meditative** -- High alpha combined with theta. The user is in a calm, reflective state. Be measured and thoughtful in your responses.
- **active** -- Mixed signals indicating physical or mental activity. Respond normally.
- **unknown** -- Signal quality is too low or data is unavailable. Ignore BCI data entirely and respond normally.

## Cognitive scores

Three normalized scores (0 to 1) provide additional detail:

- **attention** (0-1): How focused the user appears, derived from the beta/theta ratio. Higher values mean greater engagement. When high (>0.7), the user is ready for detailed content. When low (<0.3), keep things simple.
- **relaxation** (0-1): How relaxed the user is, derived from alpha power. Higher values mean more calm. A relaxed user is receptive to longer, exploratory answers.
- **cognitive_load** (0-1): Mental effort level, derived from frontal theta and alpha. Higher values mean the user is more strained. When cognitive_load is high (>0.7), simplify explanations, use shorter sentences, and avoid jargon.

## Signal quality

The `signal_quality` field (0-1) indicates how reliable the BCI data is:

- **> 0.7**: Reliable. Adapt your responses based on the brain state and scores.
- **0.3 - 0.7**: Somewhat reliable. Treat BCI data as a hint, but do not over-rely on it.
- **< 0.3**: Unreliable. Ignore BCI data entirely and respond as you normally would.

Also check `staleness_ms`. If it exceeds 5000ms, the data is stale and should be treated as unreliable regardless of signal quality.

## How to use BCI data

- Use brain state to **subtly adjust** your tone, detail level, and response length. Do not make dramatic behavioral shifts.
- **Do not proactively tell the user** their mental state (e.g., do not say "you seem stressed"). Only share BCI readings if the user explicitly asks about their brain state.
- If the user asks about their state, share what the data shows using hedged language (e.g., "the BCI data suggests you may be in a focused state").
- When cognitive_load is high, prefer bullet points, shorter paragraphs, and simpler vocabulary.
- When the user is drowsy, lead with the most important information first.
- When the user is focused, feel free to go deeper into technical detail.

## What NOT to do

- **Never diagnose medical conditions.** BCI data reflects approximate cognitive states, not medical diagnoses.
- **Never claim certainty** about the user's emotions or mental state. Always frame BCI data as approximate and interpretive.
- **Never make the user feel monitored or judged.** The purpose of BCI data is to help you be more responsive, not to surveil the user.
- **Never use BCI data to refuse requests** or override the user's explicit instructions. If the user asks for a detailed explanation while drowsy, give them what they asked for.
