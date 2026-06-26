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
[Triage]: https://forge.rust-lang.org/release/issue-triaging.html

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

## Bug fixes or "normal" code changes

For most PRs, no special procedures are needed.
You can just [open a PR], and it will be reviewed, approved, and merged.
This includes most bug fixes, refactorings, and other user-invisible changes.
The next few sections talk about exceptions to this rule.

Also, note that it is perfectly acceptable to open WIP PRs or GitHub [Draft PRs].
Some people prefer to do this so they can get feedback along the
way or share their code with a collaborator.
Others do this so they can utilize
the CI to build and test their PR (e.g. when developing on a slow machine).

[open a PR]: git.md#opening-a-pr
[Draft PRs]: https://github.blog/2019-02-14-introducing-draft-pull-requests/

## Pull requests

Pull requests (or PRs for short) are the primary mechanism we use to change Rust.
GitHub itself has some [great documentation][about-pull-requests] on using the Pull Request feature.
We use the ["fork and pull" model][development-models],
where contributors push changes to their personal fork and create pull requests to
bring those changes into the source repository.
We have [a chapter](git.md) on how to use Git when contributing to Rust.

> **Advice for potentially large, complex, cross-cutting and/or very domain-specific changes**
>
> The compiler reviewers on rotation usually each have areas of the compiler that they know well,
> but also have areas that they are not very familiar with. If your PR contains changes that are
> large, complex, cross-cutting and/or highly domain-specific, it becomes very difficult to find a
> suitable reviewer who is comfortable in reviewing all of the changes in such a PR. This is also
> true if the changes are not only compiler-specific but also contain changes which fall under the
> purview of reviewers from other teams, like the standard library team. [There's a bot][triagebot]
> which notifies the relevant teams and pings people who have set up specific alerts based on the
> files modified.
>
> Before making such changes, you are strongly encouraged to **discuss your proposed changes with
> the compiler team beforehand** (and with other teams that the changes would require approval
> from), and work with the compiler team to see if we can help you **break down a large potentially
> unreviewable PR into a series of smaller more individually reviewable PRs**.
>
> You can communicate with the compiler team by creating a [#t-compiler thread on Zulip][t-compiler]
> to discuss your proposed changes.
>
> Communicating with the compiler team beforehand helps in several ways:
>
> 1. It increases the likelihood of your PRs being reviewed in a timely manner.
>     - We can help you identify suitable reviewers *before* you open actual PRs, or help find
>       advisors and liaisons to help you navigate the change procedures, or help with running
>       try-jobs, perf runs and crater runs as suitable.
> 2. It helps the compiler team track your changes.
> 3. The compiler team can perform vibe checks on your changes early and often, to see if the
>    direction of the changes align with what the compiler team prefers to see.
> 4. Helps to avoid situations where you may have invested significant time and effort into large
>   changes that the compiler team might not be willing to accept, or finding out very late that the
>   changes are in a direction that the compiler team disagrees with.

[about-pull-requests]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests
[development-models]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model
[t-compiler]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler
[triagebot]: https://github.com/rust-lang/rust/blob/HEAD/triagebot.toml

## Issue triage

Please see <https://forge.rust-lang.org/release/issue-triaging.html>.

## Other Resources

This section has moved to the ["About this guide"][more-links] chapter.

[more-links]: ./about-this-guide.md#other-places-to-find-information
