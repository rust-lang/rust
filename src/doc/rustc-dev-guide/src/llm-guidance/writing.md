# Writing LLM-created code

LLMs are a valuable tool, but one that is easy to misuse.
The main risks are **overwhelming volume** and **lack of understanding**.
When writing your PR, keep in mind that there is a person on the other end who needs to review and understand your change,
and that other people in the project will need to read your code for years to come.
Help us out by making your change small, targeted, and high-quality.

Keep in mind this quote:

> Programs must be written for people to read, and only incidentally for machines to execute.
> — Harold Abelson and Gerald Jay Sussman, [*Structure and Interpretation of Computer Programs*][sicp]

[sicp]: https://web.mit.edu/6.001/6.037/sicp.pdf
[#llm-mentoring]: https://rust-lang.zulipchat.com/join/rlfvpemsaacs3pfi6kwqnqjb/

## Rules

### Before you write code

Before anything else, find a reviewer who volunteers to review your PR.
If you do not know where to find a reviewer, ask in [#llm-mentoring] on Zulip.
Your first message should say:

- your relevant experience, so we can find an issue that's suitable for you
- which problem (or kind of problem) you want to work on
- (optional) ideas you have so far for a solution
- (optional) how you expect to test your solution

Mentors are here to help.
Talking to them early helps you avoid wasted work.

### While working

Write your own doc-comments, `// SAFETY` comments, diagnostic wording, and soundness-critical code.
As before, you can use an LLM to review your work, but not to write it from scratch.
If you don't know what counts as soundness-critical, discuss it with your reviewer.

### When opening a PR

Disclose your use of LLMs, following the disclosure guidelines below.
Write the disclosure yourself.
You may use an LLM to privately review a disclosure you have written, but not to draft or rewrite it.

**Write your own PR description and comments**.
We want to hear from you, not from your agent.
LLM-created PR descriptions are banned.
LLM-created Github comments are banned.

## Guidelines

### Before you write code

Start with one PR at a time.
Keep your changes small enough that you and your reviewer can understand every part of them.
Go slow.

Do not use an LLM for `E-easy` issues; those are meant for you to write the code yourself.
Ask first before working on an `E-mentor` issue; mentors may not want to work with LLM-generated code.

Determine whether this is a *useful* and *well-scoped* change.
For example:

- Search for related issues and PRs.
- Find relevant code, tests, git history, and Zulip discussion.
- If this is a cross-cutting change, consult the [pull request guidance](../getting-started.md#pull-requests).
- [Make the smallest change that fixes the problem][small-cls].
  Do not combine it with unrelated refactors or cleanups.

[small-cls]: https://google.github.io/eng-practices/review/developer/small-cls.html

### While working

When fixing a bug, verify that your test fails before and succeeds after your change.
Consult [adding new tests](../tests/adding.md) and [best practices](../tests/best-practices.md) for test procedures.
Tests are absolutely required; either existing tests or new tests you write.
Untested LLM PRs will not be merged.

Mass renames or rewrites should *strongly* prefer using a proper syntax rewrite tool, such as [`ast-grep`].
You may use an LLM for generating the instructions for that tool, but you should be very cautious about performing the rewrite directly with an LLM.

Consider [performance] as you write.

[performance]: ../contributing.md#performance

Think before adding dependencies;
consult our [guidance for new dependencies][crates-io].

[crates-io]: ../crates-io.md

Verify your understanding against the existing code, documentation, and tests.
You can get better advice from your LLM by telling *it* to read the relevant materials.
Do not rely on the LLM as a source of truth.

### Write maintainable code

Treat generated code as a *draft*, not a final product.
Follow our [correctness and maintainability conventions](../conventions.md#cc).

Avoid unnecessary abstractions and compatibility layers.
Rustc does not have a stable API; you do not need to preserve backwards compatibility for internal compiler APIs.

### Commit structure

See ["How to structure your PR"](../contributing.md#er).
Commit messages must be authored by you, not your LLM.

### Before opening a PR

Review your own PR before opening it: Does it make sense?
Can you tell what the goal of the PR is?
Does it achieve that goal?

Remove outdated or prototyping code and debugging.

Re-read the whole diff, *not* just your conversation with the agent.
Your reviewer is going to see your code, not your conversation.

[Run tests](../tests/running.md) to verify your change works.
Do NOT report which UI tests you ran in the PR description;
that's noise, since CI will run them anyway.
If you did manual testing or benchmarking, do report that,
but note that all LLM PRs must have automated tests.

[Review diagnostic snapshots](../tests/adding.md#step-4-review-the-output);
don't simply `--bless` them away.

We recommend using a different model for adversarial local review before publishing your changes.
You're still responsible for reviewing your own changes yourself.

### Understand your own change

We want you to understand and be able to explain your own change and its edge cases.
Asking the LLM can be a starting point but it's not the same as explaining it yourself.

Try explaining your change to yourself before opening the PR.
For example, ask yourself:

- What is the original bug?
  When does it happen?
  How severe is it?
  What causes it?
- Why is this the right fix?
  Are there other fixes possible?
  What are their advantages or disadvantages?
- Are there any edge cases?
  Does your code handle them?
- Does the code have existing [invariants](https://brooker.co.za/blog/2023/07/28/ds-testing.html)?
  Did you preserve those invariants?
- What behavior is *unchanged*?
  What test establishes that?
- Why does your test trigger the bug?
- What are you still not certain about?

It's ok to be uncertain and to ask for help.
We would much rather help you because you're not sure than have you guess wrong and then have to reverse-engineer where you went wrong.

[`ast-grep`]: https://astgrep.com/

### Disclosure guidelines

Disclose the *extent* and *purpose* of your LLM use.
We don't care which model you used, but we do care whether you used the LLM to implement the idea or to come up with it.

**Good** examples:

> LLM disclosure: I wrote the three commits by hand after viewing profiling data. I used an LLM to review the commits before submitting. The LLM identified that `ImplString::is_negative` was no longer used, so I removed that field by hand.

> Created with the help of an LLM, which:
> - traced the missing cache hits to the unconditional `return(pass)` by inspecting Fastly vs CloudFront headers,
> - reviewed the git history to understand why the snippet was added, and
> - made the VCL change.

**Bad** examples:

> 🤖 Generated with Claude Code

> `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`
