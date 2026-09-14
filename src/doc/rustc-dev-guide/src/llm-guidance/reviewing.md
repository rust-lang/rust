# Reviewing with LLMs

## Using an LLM to review code

- If a more reliable tool, such as a linter or formatter, already exists for the language you're writing, we strongly suggest using that tool instead of or in addition to the LLM.
- Configure LLM review tools to reduce false positives and excessive focus on trivialities, as these are common, exhausting failure modes.
- Wherever possible, ask an LLM to *generate or configure a linter*, which you then tell it to run.
  This both saves on token costs, and allows people who are not using an LLM to run the analysis.
  For example, if you have a codebase-specific wrapper around command spawning,
  rather than getting an LLM to look for places where you should use the wrapper,
  [configure clippy to disallow `Command::new`][disallowed-methods].
- LLMs sometimes prefer LLM-generated output, particularly output from the same model.
  Treat LLM review as advisory, and do not rely on the model that
  produced a change as its only reviewer.

[disallowed-methods]: https://doc.rust-lang.org/clippy/lint_configuration.html#disallowed-methods

## Reviewing LLM-created code

First, add the new `llm-assisted` label to the PR.

### Rules

We expect everyone to follow the new policy, not just authors.
That means it is **your responsibility** to check whether an `llm-assisted` PR touches an area that's disallowed by the policy.
You may request that the author redo it without LLM-generated code, in which case this section doesn't apply.

The following areas are currently banned:
- Code that affects soundness.
  If the author is not an org member who is experienced in the domain, you are required to close the PR.
- Diagnostics.
  All user-facing diagnostics must be human-written.
- Docs.
  All public doc-comments, and all `// SAFETY` comments, must be human-written.

"Code that affects soundness" is both broader and narrower than it sounds.
It's broader because almost all of the compiler is relevant to soundness;
it's narrower because there's quite a lot of rust-lang/rust that isn't the compiler
(library, bootstrap, compiletest, rustdoc, CI, ...).

If in doubt, we suggest this criteria:
Do not allow LLM-generated code for parts of the compiler where [wrong code does not look wrong][joel-wrong].
Ultimately, this is up to your judgement as a reviewer.

[joel-wrong]: https://www.joelonsoftware.com/2005/05/11/making-wrong-code-look-wrong/

You are still expected to respect your [r+ rights](../compiler-team.md#r-rights).
Please do not merge PRs unless you are confident in that part of that code,
even if the maintainer does not wish to review LLM PRs.

### Guidelines

Point people to [#llm-mentoring] liberally.
Deal with low-quality PRs by closing the PR and asking the author to follow the policy.
Deal with borderline PRs by asking the author to put in the work themselves rather than offloading it to you.
For example, ask them to reproduce the bug, explain the change in their own
words, identify relevant edge cases, or add or justify tests.

If you find yourself suggesting the same fixes on multiple PRs,
consider adding them to the dev-guide.

#### Missing disclosure

If you see a PR that is "obviously" LLM-created without disclosing that use, you have the option—but not the responsibility—to close it unilaterally.

We suggest using the following wording:

```markdown
This PR appears to be LLM-generated without disclosing use of an LLM, so I am going to close this PR.
You are welcome to open additional PRs as long as they follow our [policy][forge-page].
For more information, see [#llm-mentoring] on Zulip.

[#llm-mentoring]: https://rust-lang.zulipchat.com/join/rlfvpemsaacs3pfi6kwqnqjb/
[forge-page]: https://forge.rust-lang.org/policies/llm-usage.html
```

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

```markdown
You've opened an LLM-generated PR, but it's in the normal review queue, which breaks our [policy][experiment guidelines].
I am going to close this PR.
Please do not re-open it until you find a project member who has volunteered to review it.
For more information, see [#llm-mentoring] on Zulip.

[#llm-mentoring]: https://rust-lang.zulipchat.com/join/rlfvpemsaacs3pfi6kwqnqjb/
[forge-page]: https://forge.rust-lang.org/policies/llm-usage.html
```

#### Missing tests, low-quality, or not self-reviewed

If a PR is clearly not ready for review, you do not have to review it.
It's ok to simply skim the PR and tell the author "you need to add tests before I can review this".
If you notice on your skim that the PR is clearly the wrong approach, it's ok to close the PR and tell the author to talk with you in the [#llm-mentoring] channel before opening a new PR.

[experiment guidelines]: https://forge.rust-lang.org/policies/llm-usage.html#experiment-llm-created-code-changes-intended-for-review
[#llm-mentoring]: https://rust-lang.zulipchat.com/join/rlfvpemsaacs3pfi6kwqnqjb/
[forge-page]: https://forge.rust-lang.org/policies/llm-usage.html
