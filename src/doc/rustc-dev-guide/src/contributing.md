# Contribution procedures

## Bug reports

While bugs are unfortunate, they're a reality in software.
We can't fix what we don't know about, so please report liberally.
If you're not sure if something is a bug, feel free to open an issue anyway.

**If you believe reporting your bug publicly represents a security risk to Rust users,
please follow our [instructions for reporting security vulnerabilities][vuln]**.

[vuln]: https://www.rust-lang.org/policies/security

If you're using the nightly channel, please check if the bug exists in the
latest toolchain before filing your bug.
It might be fixed already.

If you have the chance, before reporting a bug, please [search existing issues],
as it's possible that someone else has already reported your error.
This doesn't always work, and sometimes it's hard to know what to search for, so consider this
extra credit.
We won't mind if you accidentally file a duplicate report.

Similarly, to help others who encountered the bug find your issue, consider
filing an issue with a descriptive title, which contains information that might be unique to it.
This can be the language or compiler feature used, the
conditions that trigger the bug, or part of the error message if there is any.
An example could be: **"impossible case reached" on lifetime inference for impl
Trait in return position**.

Opening an issue is as easy as following [this link][create an issue] and filling out the fields
in the appropriate provided template.

[search existing issues]: https://github.com/rust-lang/rust/issues?q=is%3Aissue
[create an issue]: https://github.com/rust-lang/rust/issues/new/choose

## Other procedures

Other contribution procedures are documented with the parts of the guide that own them:

<a id="bug-fixes-or-normal-code-changes"></a>
<a id="pull-requests"></a>

- For routine changes or large, cross-cutting pull requests, see [normal code changes](getting-started.md#bug-fixes-or-normal-code-changes), [PR lifecycle](./pr-lifecycle.md), and [pull request guidance](getting-started.md#pull-requests).

<a id="new-features"></a>
<a id="breaking-changes"></a>
<a id="major-changes"></a>

- For new features, major changes, and breaking changes, see [Implementing new language features](implementing-new-features.md) and [Procedures for breaking changes](bug-fix-procedure.md).

<a id="performance"></a>

- For performance-sensitive changes, see [Performance testing](tests/perf.md#performance-considerations).

<a id="writing-documentation"></a>

- For compiler documentation, see [Contributing documentation](building/compiler-documenting.md#contributing-documentation).

<a id="issue-triage"></a>

- For issue triage, see [Issue triage](getting-started.md#issue-triage).

<a id="llm-guidance"></a>

- For guidance on LLM usage, see [Running LLMs](llm-guidance.md).

<a id="contributing-to-rustc-dev-guide"></a>

- For changes to this guide itself, see [Writing rustc-dev-guide documentation](contributing-to-guide.md).

<a id="external-dependencies"></a>

- For external dependencies, see [Using external repositories](external-repos.md).

<a id="helpful-links-and-information"></a>

- For further resources, see [About this guide](about-this-guide.md#other-places-to-find-information).
