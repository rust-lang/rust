# Running LLMs

This is a non-binding list of suggestions for working with LLMs.
This is not our moderation policy; see [Forge][LLM policy].

## Review bots

- If a more reliable tool, such as a linter or formatter, already exists for the language you're writing, we strongly suggest using that tool instead of or in addition to the LLM.
- Configure LLM review tools to reduce false positives and excessive focus on trivialities, as these are common, exhausting failure modes.
- Wherever possible, ask an LLM to *generate a linter*, which you then tell it to run.
  This both saves on token costs, and allows people who are not using an LLM to run the analysis.

## LLM-authored code

- We recommend, but do not require, using a second LLM for adversarial local review before publishing your changes.
- Mass renames or rewrites should *strongly* prefer using a proper syntax rewrite tool, such as [`ast-grep`].
  You may use an LLM for generating the instructions for that tool, but you should be very cautious about performing the rewrite directly with an LLM.

[LLM policy]: https://forge.rust-lang.org/policies/llm-usage.html
