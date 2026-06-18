# Working with LLMs

This is a non-binding list of suggestions for working with LLMs.
This is not our moderation policy; see [Forge][LLM policy].

[LLM policy]: https://forge.rust-lang.org/policies/llm-usage.html

## Automated checks and LLM review

- If a more reliable tool, such as a linter or formatter, already exists for the language you're writing, we strongly suggest using that tool instead of or in addition to the LLM.
- Configure LLM review tools to reduce false positives and excessive focus on trivialities, as these are common, exhausting failure modes.
- Wherever possible, ask an LLM to *generate a linter*, which you then tell it to run.
  This both saves on token costs, and allows people who are not using an LLM to run the analysis.
- LLMs sometimes prefer LLM-generated output, particularly output from the same
  model. Treat LLM review as advisory, and do not rely on the model that
  produced a change as its only reviewer.

## Writing LLM-created code

- We recommend, but do not require, using a different model for adversarial
  local review before publishing your changes. This does not replace human
  self-review.
- Mass renames or rewrites should *strongly* prefer using a proper syntax rewrite tool, such as [`ast-grep`].
  You may use an LLM for generating the instructions for that tool, but you should be very cautious about performing the rewrite directly with an LLM.

[`ast-grep`]: https://astgrep.com/

## Reviewing LLM-created code

Point people to [#llm-mentoring](https://rust-lang.zulipchat.com/#narrow/channel/606558-llm-mentoring/) liberally.
Deal with low-quality PRs by closing the PR and asking the author to follow the "solicited" rule in the Forge policy.
Deal with borderline PRs by asking the author to do work that shows they're paying attention; it's ok to ask for that work before you've put much time into review yourself.
For example, ask them to reproduce the bug, explain the change in their own
words, identify relevant edge cases, or add or justify tests.

If you find yourself suggesting the same fixes on multiple PRs,
consider adding them to the dev-guide.

## Disclosure guidelines

Disclose the *extent* and *purpose* of your LLM use.
We don't care which model you used, but we do care whether you used the LLM to
implement the idea or to come up with it.
Write the disclosure yourself. You may use an LLM to privately review a
disclosure you have written, but not to draft or rewrite it.

**Good** examples:

> LLM disclosure: I wrote the three commits by hand after viewing profiling data. I used an LLM to review the commits before submitting. The LLM identified that ImplString::is_negative was no longer used, so I removed that field by hand.

> Created with the help of Claude Code, which:
> - traced the missing cache hits to the unconditional return(pass) by inspecting Fastly vs CloudFront headers,
> - reviewed the git history to understand why the snippet was added, and
> - made the VCL change.

**Bad** examples:

> 🤖 Generated with Claude Code

> Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
