# Working with LLMs

This is a list of guidelines for working with LLMs, as well as a summary of the moderation policy.
This is not the policy itself; see [Forge][LLM policy].
If the two conflict, Forge is canonical.

[LLM policy]: https://forge.rust-lang.org/policies/llm-usage.html

## Writing LLM-created code

### Rules

Before anything else, you **must** find a reviewer who volunteers to review your PR.
LLM-created PRs without an assigned reviewer will be closed.
If you do not know where to find a reviewer, see [#llm-mentoring] on Zulip.

Disclose your use of LLMs, following the disclosure guidelines below.
Write the disclosure yourself.
You may use an LLM to privately review a disclosure you have written, but not to draft or rewrite it.

You may *not* use an LLM to write doc-comments, `// SAFETY` comments, diagnostic wording, or soundness-critical code.
Write them yourself instead.
If you don't know what counts as soundness-critical, discuss it with your reviewer.

**Write your own PR description and comments**.
LLM-created PR descriptions are banned.
LLM-created Github comments are banned.
We want to hear from you, not from your agent.

### Guidelines

#### Before you write code

Start with one PR at a time.
We understand it's easy to be excited about contributing to Rust, but your PRs are not only a gift but a responsibility for the reviewers.
Go slow.

Avoid E-easy and E-mentor issues.
Those are intended for people to get familiar with the project, not for LLMs.

Determine whether this is a *useful* and *well-scoped* change.
For example:

- Search for related issues and PRs.
- Find relevant code, tests, git history, and Zulip discussion.
- If this is a cross-cutting change, consult the "cross-cutting" section of [the contributing docs](./contributing.md#pull-requests).
- Make the smallest change that fixes the problem.
  Do not combine it with unrelated refactors or cleanups.

#### While working

When fixing a bug, verify that your test fails before and succeeds after your change.
Consult [adding new tests](./tests/adding.md) and [best practices](./tests/best-practices.md) for test procedures.
Tests are absolutely required; either existing tests or new tests you write.
Untested LLM PRs will not be merged.

Mass renames or rewrites should *strongly* prefer using a proper syntax rewrite tool, such as [`ast-grep`].
You may use an LLM for generating the instructions for that tool, but you should be very cautious about performing the rewrite directly with an LLM.

#### Before opening a PR

Review your own PR before opening it:
Does it make sense? Can you tell what the goal of the PR is? Does it achieve that goal?

[Run tests](tests/running.md) to verify your change works.
Do NOT report which tests you ran in the PR description;
that's useless to us, since CI will run them anyway.

You must [review diagnostic snapshots](tests/adding.md#step-4-review-the-output);
don't simply `--bless` them away.

We recommend using a different model for adversarial local review before publishing your changes.
This does not replace human self-review.

#### Understand your own change

You must understand and be able to explain your own change and its edge cases.
Asking the LLM can be a starting point but is not sufficient.
You are responsible for your own code; you cannot disclaim responsibility to your agent.

Some questions that you should be able to answer:

- What is the original bug? When does it happen? How severe is it? What causes it?
- Why is this the right fix? Are there other fixes possible? What are their advantages or disadvantages?
- Are there any edge cases? Does your code handle them?
- What behavior is *unchanged*? What test establishes that?
- Why does your test trigger the bug?
- What are you still not certain about?

[`ast-grep`]: https://astgrep.com/

#### Disclosure guidelines

Disclose the *extent* and *purpose* of your LLM use.
We don't care which model you used, but we do care whether you used the LLM to implement the idea or to come up with it.

**Good** examples:

> LLM disclosure: I wrote the three commits by hand after viewing profiling data. I used an LLM to review the commits before submitting. The LLM identified that `ImplString::is_negative` was no longer used, so I removed that field by hand.

> Created with the help of Claude Code, which:
> - traced the missing cache hits to the unconditional `return(pass)` by inspecting Fastly vs CloudFront headers,
> - reviewed the git history to understand why the snippet was added, and
> - made the VCL change.

**Bad** examples:

> 🤖 Generated with Claude Code

> Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

## Automated checks and LLM review

- If a more reliable tool, such as a linter or formatter, already exists for the language you're writing, we strongly suggest using that tool instead of or in addition to the LLM.
- Configure LLM review tools to reduce false positives and excessive focus on trivialities, as these are common, exhausting failure modes.
- Wherever possible, ask an LLM to *generate a linter*, which you then tell it to run.
  This both saves on token costs, and allows people who are not using an LLM to run the analysis.
- LLMs sometimes prefer LLM-generated output, particularly output from the same
  model. Treat LLM review as advisory, and do not rely on the model that
  produced a change as its only reviewer.

## Reviewing LLM-created code

First, add the new `ai-assisted` label to the PR.

### Rules

We expect everyone to follow the new policy, not just authors.
That means it is **your responsibility** to check whether an `ai-assisted` PR touches an area that's disallowed by the policy.
You may request that the author redo it without LLM-generated code, in which case this section doesn't apply.

The following areas are currently banned:
- Code that affects soundness. If the author is not an org member who is experienced in the domain, you are required to close the PR.
- Diagnostics. All user-facing diagnostics must be human-written.
- Docs. All public doc-comments, and all `SAFETY` comments, must be human-written.

### Guidelines

Point people to [#llm-mentoring] liberally.
Deal with low-quality PRs by closing the PR and asking the author to follow the policy.
Deal with borderline PRs by asking the author to do work that shows they're paying attention; it's ok to ask for that work before you've put much time into review yourself.
For example, ask them to reproduce the bug, explain the change in their own
words, identify relevant edge cases, or add or justify tests.

If you find yourself suggesting the same fixes on multiple PRs,
consider adding them to the dev-guide.

#### Missing disclosure

If you see a PR that is "obviously" LLM-created without disclosing that use, you have the option—but not the responsibility—to close it unilaterally.

We suggest using the following wording:

> This PR appears to be LLM-generated without disclosing use of an LLM,
> so I am going to close this PR.
> You are welcome to open additional PRs as long as they follow our [policy][forge-page].
> For more information, see [#llm-mentoring] on Zulip.

Examples of "obvious" LLM tells are:
- PR descriptions that are completely wrong/don't match the code.
- PR descriptions that state the exact tests that were run (e.g. `./x test --stage 1 tests/ui/<the-new-test-name>.rs`) or useless tests such as `git diff --check`.
- Responses to reviewer questions that fall into one of the above categories.

You do not have an obligation to detect LLM-created PRs;
you don't need to play detective.

PR templates will have a "Did you use an LLM?" question so that this rarely comes up.
If the author deleted the question without answering it, you can close the PR, no questions asked.

#### Missing solicited reviewer

If the PR discloses use, but does not assign a reviewer following the [experiment guidelines], you can close it similarly:

> You've opened an LLM-generated PR, but it's in the normal review queue, which breaks our [policy][experiment guidelines].
> I am going to close this PR.
> Please do not re-open it until you find a project member who has volunteered to review it.
> For more information, see [#llm-mentoring] on Zulip.

#### Missing tests, low-quality, or not self-reviewed

If a PR is clearly not ready for review, you do not have to review it.
It's ok to simply skim the PR and tell the author "you need to add tests before I can review this".
If you notice on your skim that the PR is clearly the wrong approach, it's ok to close the PR and tell the author to talk with you in the [#llm-mentoring] channel before opening a new PR.

[experiment guidelines]: https://forge.rust-lang.org/policies/llm-usage.html#experiment-llm-created-code-changes-intended-for-review
[#llm-mentoring]: https://rust-lang.zulipchat.com/join/rlfvpemsaacs3pfi6kwqnqjb/
[forge-page]: https://forge.rust-lang.org/policies/llm-usage.html
