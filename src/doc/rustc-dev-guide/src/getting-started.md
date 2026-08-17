# Getting Started

Thank you for your interest in contributing to Rust!
There are many ways to contribute, and we appreciate all of them.

For general information about how to contribute to Rust,
see [Forge](https://forge.rust-lang.org/how-to-start-contributing.html).
The rest of this section is about how to contribute to the compiler specifically.

This documentation is _not_ intended to be comprehensive;
it is meant to be a quick guide for the most useful things.
For more information,
see [How to build and run the compiler](building/how-to-build-and-run.md).

## Finding help

See also ["Asking
Questions"](https://forge.rust-lang.org/how-to-start-contributing.html#asking-questions).

### Experts

Not all `t-compiler` members are experts on all parts of `rustc`;
it's a pretty large project.
To find out who could have some expertise on
different parts of the compiler, [consult triagebot assign groups][map].
The sections that start with `[assign*` in `triagebot.toml` file.
But also, feel free to ask questions even if you can't figure out who to ping.

Another way to find experts for a given part of the compiler is to see who has made recent commits.
For example, to find people who have recently worked on name resolution since the 1.68.2 release,
you could run `git shortlog -n 1.68.2.. compiler/rustc_resolve/`.
Ignore any commits starting with
"Rollup merge" or commits by `@bors` (see [CI contribution procedures](./contributing.md#ci) for
more information about these commits).

[map]: https://github.com/rust-lang/rust/blob/HEAD/triagebot.toml


## What should I work on?

The `rust` monorepo is quite large and it can be difficult to know which parts need
help, or are a good starting place for beginners.
Here are some suggested starting places.

### Easy or mentored issues

If you're looking for somewhere to start, check out the following [issue
search][help-wanted-search].
See the [Triage] for an explanation of these labels.
You can also try filtering the search to areas you're interested in.
For example:

- `label:T-compiler` will only show issues related to the compiler
- `label:A-diagnostics` will only show diagnostic issues

Not all important or beginner work has issue labels.
See below for how to find work that isn't labelled.

[help-wanted-search]: https://github.com/rust-lang/rust/issues?q=is%3Aopen%20is%3Aissue%20org%3Arust-lang%20no%3Aassignee%20label%3AE-easy%2CE-medium%2CE-help-wanted%2CE-mentor%20-label%3AS-blocked%20-linked%3Apr
[Triage]: ./contributing.md#issue-triage

### Recurring work

Some work is too large to be done by a single person.
In this case, it's common to have "Tracking issues" to co-ordinate the work between contributors.
Here are some example tracking issues where
it's easy to pick up work without a large time commitment:

- *Add recurring work items here.*

If you find more recurring work, please feel free to add it here!

### Diagnostic issues

Many diagnostic issues are self-contained and don't need detailed background knowledge of the
compiler.
You can see a list of diagnostic issues [here][diagnostic-issues].

[diagnostic-issues]: https://github.com/rust-lang/rust/issues?q=is%3Aissue+is%3Aopen+label%3AA-diagnostics+no%3Aassignee

### Picking up abandoned pull requests

Sometimes, contributors send a pull request, but later find out that they don't have enough
time to work on it, or they simply are not interested in it anymore.
Such PRs are often eventually closed and they receive the `S-inactive` label.
You could try to examine some of these PRs and pick up the work.
You can find the list of such PRs [here][abandoned-prs].

If the PR has been implemented in some other way in the meantime, the `S-inactive` label
should be removed from it.
If not, and it seems that there is still interest in the change,
you can try to rebase the pull request on top of the latest `main` branch and send a new
pull request, continuing the work on the feature.

[abandoned-prs]: https://github.com/rust-lang/rust/pulls?q=is%3Apr+label%3AS-inactive+is%3Aclosed

### Writing tests

Issues that have been resolved but do not have a regression test are marked with the `E-needs-test` label.
Writing unit tests is a low-risk,
lower-priority task that offers new contributors a great opportunity to familiarize themselves
with the testing infrastructure and contribution workflow.
You can see a list of needs test issues [here][needs-test-issues].

[needs-test-issues]: https://github.com/rust-lang/rust/issues?q=is%3Aissue%20is%3Aopen%20label%3AE-needs-test%20no%3Aassignee

### Contributing to std (standard library)

See [std-dev-guide](https://std-dev-guide.rust-lang.org/).

### Other ways to contribute

See [Forge](https://forge.rust-lang.org/how-to-start-contributing.html#how-to-start-contributing-1).

## Cloning and Building

See ["How to build and run the compiler"](./building/how-to-build-and-run.md).

## Contributor Procedures

This section has moved to the ["Contribution Procedures"](./contributing.md) chapter.

## Other Resources

This section has moved to the ["About this guide"][more-links] chapter.

[more-links]: ./about-this-guide.md#other-places-to-find-information
