# Running LLMs

This is a non-binding list of suggestions for working with LLMs.
This is not our moderation policy; see [Forge][LLM policy].

## Review bots

- If a more reliable tool, such as a linter or formatter, already exists for the language you're writing, we strongly suggest using that tool instead of or in addition to the LLM.
- Configure LLM review tools to reduce false positives and excessive focus on trivialities, as these are common, exhausting failure modes.

## LLM-authored code

- We recommend, but do not require, using a second LLM for adversarial local review before publishing your changes.

[LLM policy]: https://forge.rust-lang.org/policies/llm-usage.html
