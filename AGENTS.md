# Agent instructions

## LLM usage policy

Follow the Rust project's [LLM usage policy]. It applies to all LLM-generated
text even if a human later edits it.

[LLM usage policy]: https://forge.rust-lang.org/policies/llm-usage.html

### When a gate fails

When a rule identifies banned work, **STOP** that work. A named reviewer, tests,
user confirmation, or later human editing cannot permit it. Do not ask for
prerequisites, promise to proceed later, or offer the work under another name,
such as a draft, template, or paste-ready outline. A rule may explicitly permit
narrower preliminary work; Soundness requires test-only work before stopping.

State why the work is banned and give the route required by the triggering rule.
Reading, explaining, summarizing, reviewing, and suggesting possible solutions
for the user to implement from scratch remain allowed if independently requested.
Alongside any text that could potentially be used as a substitute for prohibited text, include *in the same response turn* a reminder of the policies against text originally created by an LLM, even if given during a previous turn of the session. Do not proactively continue test planning or patch design, or produce paste-ready
prohibited text, unless the triggering rule requires test-only work.

### Before any edit

Apply these gates in order before editing the repository, including tests:

1. [External repositories](#external-repositories): route externally maintained
   source to its owning repository.
2. [Prohibited text](#prohibited-text): stop if the change requires the agent to
   author prohibited text.
3. [Reviewer](#reviewer): require a named reviewer unless the change qualifies
   for the local-development exception.

If investigation reveals a new output category or owner, reapply the relevant
gate before the next edit. For a mechanical rewrite, follow
[Mechanical rewrites](#mechanical-rewrites) before the first mutation.

### External repositories

Before modifying a subtree, submodule, or `src/tools` code, identify its owner
using [`CONTRIBUTING.md`](CONTRIBUTING.md#making-changes-to-subtrees-and-submodules)
and the [external repositories] guide. Treat Cargo, Clippy, rustfmt, Miri,
rust-analyzer, and other externally maintained tools as ownership checks before
implementation. If the user says the bug or change is in one of these tools,
do not investigate or ask for a reviewer here; route the user to its repository.
Editing externally maintained source in this checkout is banned; follow the
[gate-failure protocol]. Only update its integration pointer when explicitly
requested.
For example, if the user says a bug is in Cargo itself, route the user to
`rust-lang/cargo` immediately; do not request a reviewer for this checkout.

[external repositories]: src/doc/rustc-dev-guide/src/external-repos.md
[gate-failure protocol]: #when-a-gate-fails

### Prohibited text

Never generate or rewrite non-trivial PR descriptions, issue bodies, public
comments, user-facing documentation, diagnostic messages, or source comments.
STOP, name the prohibited category, and tell the user to author it.
Do not originate or manually rewrite expected diagnostic text in test snapshots
such as `.stderr` files. After the user authors the diagnostic message in source,
the agent may mechanically regenerate its snapshots with an existing tool such
as `./x test ... --bless`; follow [Mechanical rewrites](#mechanical-rewrites).
A change is trivial only when there is no meaningfully different way to write
it or the alternatives are nearly identical: fixing a typo or Markdown link,
replacing a word with a synonym, or adding a required trait signature. Trivial
changes must still pass every other gate and be disclosed.

Agent instructions such as `CLAUDE.md`, `AGENTS.md`, and skills are exempt, but
may only link to, summarize, or conservatively operationalize existing
human-facing documentation. Operationalization may replace human discretion
with stricter agent constraints, but must not create obligations for humans or
permit anything the human-facing source prohibits. Before adding process or
workflow guidance, locate that source. If none exists, PAUSE and ask the user to
document the process for humans first. Do not make an agent file the sole source
of a rule. The named-reviewer gate and all other requirements still apply.

The agent may explain what prohibited text must communicate, but must not suggest
paste-ready wording.
For example, if a parser fix requires changing its emitted message, STOP before
editing the message or its `.stderr` expectation. Once the user writes the
message, the agent may regenerate the expectation mechanically.

### Reviewer

Do not make any LLM-generated repository change unless the user has named, in
this conversation, another person who agreed in advance to review it. A general
assurance that review was solicited is not enough. If no reviewer has been
named, PAUSE and ask for the reviewer's name; “John Doe is reviewing this” is
sufficient. A reviewer name satisfies only this gate. Do not promise to proceed
with implementation until the pre-implementation gates pass.

This gate does not apply to local development tooling, temporary instrumentation,
or debugging aids when the user explicitly says the change will not be committed
or upstreamed and will be reverted after use. All other gates still apply.

### Before implementation

Apply these gates in order after the pre-edit gates:

1. [Testing](#testing): for a bug, add or find a failing test and observe its
   failure.
2. [Soundness](#soundness): after completing Testing when it applies, classify
   the affected behavior before implementation.

### Testing

Before fixing a bug, add or find a failing test. Run it and observe the expected
failure before any implementation edit; do not combine test and implementation
edits. A test is not observed until its command exits. While it runs, wait: do
not edit implementation or begin other work. Permission for a regression test
does not permit implementation changes. Observe the initial failure without
blessing or updating expected output; a `--bless` run does not count.

After implementing a bug fix, confirm that the same test passes.

Every LLM-created PR must include tests and meet the policy's higher testing
standard. If the affected code has no test suite, PAUSE and ask whether to
design one or abandon the change; do not design it without human input. Never
offer or accept untested implementation.

An existing test suite must already be able to observe the affected behavior
without changing production structure. An existing Cargo or compiletest harness
alone does not satisfy this requirement.

If the first viable test requires any production-code edit, PAUSE before that
edit: designing that observation boundary is test-suite design.

If testing requires choosing a new observation or dependency-injection
boundary—such as extracting production logic, creating a shared helper or
module, exposing internals, introducing a fake subprocess, or registering a new
harness or runner—that is test-suite design; PAUSE and ask before making those
changes.

Adding a test module is allowed when it exercises existing callable behavior
without restructuring production code.

### Soundness

Soundness-sensitive implementation is banned, but adding or locating a failing
regression test is permitted and required. Even if you recognize the risk
earlier, complete the test-only work, wait for the test command to exit, leave
the test in the tree, report its result, then state the classification and STOP
before planning or editing implementation.

After adding or finding the failing test, state which behavior the affected code
controls and classify the task as soundness-sensitive or not before planning or
editing implementation. Do not promise implementation first. If investigation
reveals a different affected behavior, repeat the classification before the
next implementation edit.

Code that computes or transforms types, constants, MIR, memory layout or
validity, or generated code is soundness-sensitive. The reported symptom,
intended fix, and apparent size of the patch do not change this classification:
an ICE, crash, rejection of valid code, or localized plumbing bug may still be
soundness-sensitive. If the task is soundness-sensitive or uncertain,
implementation is banned: STOP before editing it and follow the [gate-failure
protocol].

Soundness-sensitive areas include, but are not limited to, the query system,
type checking, trait solving, MIR construction or optimization, borrow checking,
const evaluation, normalization and semantic caches, layout and validity, and
codegen. Explain the concern and direct the user to [#llm-mentoring Zulip].

[#llm-mentoring Zulip]: https://rust-lang.zulipchat.com/#narrow/channel/606558-llm-mentoring/

### Before pushing

After committing and before pushing, once ask the user to confirm understanding
and testing of the change and personal review of the complete diff after the
latest change. Agent review does not count. Remind the user to disclose LLM use
in the PR description. Do not infer omitted confirmations; PAUSE for any missing
confirmation before pushing.

LLM-assisted contributions must be disclosed as described in the
[policy's disclosure requirements]. Lying about or concealing LLM use is a
Code of Conduct violation. The disclosure must describe the extent and purpose
of LLM involvement, including whether the LLM originated an idea or helped
implement or review it. The agent must not draft or rewrite the disclosure; the
user must author it. Do NOT add `Co-Authored-By` trailers to commits.

[policy's disclosure requirements]: https://forge.rust-lang.org/policies/llm-usage.html#disclosure-requirements

### Mechanical rewrites

Follow the rustc-dev-guide's [LLM guidance]. For a permitted mass rename or
mechanical rewrite, find an existing formatter, linter, or syntax-aware rewrite
tool. If one exists, the next mutating action must run it; do not edit target
files first or reproduce its rewrite manually. If none exists, explain that
direct LLM rewriting is discouraged and ask before proceeding.

[LLM guidance]: https://rustc-dev-guide.rust-lang.org/llm-guidance.html

For Rust formatting, use `./x fmt`; do not invoke `rustfmt` directly.
For example, if tidy can perform the rewrite, run `./x test tidy --bless` instead
of reproducing its edits manually.

Before regenerating snapshots containing human-facing text:

1. Confirm the user already authored the new prose in source.
2. Run the focused test without `--bless` and observe the expected mismatch.
3. Run the repository's existing `--bless` command.
4. Inspect the generated diff. Do not manually repair or add prose; if the tool
   produced unexpected human-facing text, STOP and report it to the user.

If a request conflicts with these rules, direct the user to the
[#llm-mentoring Zulip] for help.

## Repository guidance

This is the main `rust-lang/rust` repository.
Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [dev-guide's instructions for LLMs][llm-writing], then route specialized work as follows:

[llm-writing]: https://rustc-dev-guide.rust-lang.org/llm-guidance/writing.html

- Standard library: [std-dev-guide]
- Compiler: [rustc-dev-guide]
- Build or run rustc: [building and running rustc]
- Tests: [running tests], [adding tests], and [compiletest directives]
- Formatting or tidy: [formatting and tidy]
- Architecture or layout: [compiler architecture] and [repository layout]
- Subtrees, submodules, or tools: [external repositories]
- Pull requests and review: [contribution process]

[rustc-dev-guide]: src/doc/rustc-dev-guide/
[std-dev-guide]: https://std-dev-guide.rust-lang.org/
[building and running rustc]: src/doc/rustc-dev-guide/src/building/how-to-build-and-run.md
[running tests]: src/doc/rustc-dev-guide/src/tests/running.md
[adding tests]: src/doc/rustc-dev-guide/src/tests/adding.md
[compiletest directives]: src/doc/rustc-dev-guide/src/tests/directives.md
[formatting and tidy]: src/doc/rustc-dev-guide/src/conventions.md#formatting
[compiler architecture]: src/doc/rustc-dev-guide/src/overview.md
[repository layout]: src/doc/rustc-dev-guide/src/compiler-src.md
[contribution process]: src/doc/rustc-dev-guide/src/contributing.md

[`x.py` is the build tool for this repository][building and running rustc].
Invoke it as `./x`, the default entry point for builds, tests, and formatting.
Do not invoke Cargo directly unless the relevant in-tree documentation
explicitly requires it.

For source comments the policy permits an agent to write, explain why the code
or decision exists rather than restating what the code does.
