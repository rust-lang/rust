# Getting Started

Thank you for your interest in contributing to Rust! There are many ways to
contribute, and we appreciate all of them.

<!-- toc -->

If this is your first time contributing, the [walkthrough] chapter can give you a good example of
how a typical contribution would go.

This documentation is _not_ intended to be comprehensive; it is meant to be a
quick guide for the most useful things. For more information, [see this
chapter on how to build and run the compiler](./building/how-to-build-and-run.md).

[internals]: https://internals.rust-lang.org
[rust-discord]: http://discord.gg/rust-lang
[rust-zulip]: https://rust-lang.zulipchat.com
[coc]: https://www.rust-lang.org/policies/code-of-conduct
[walkthrough]: ./walkthrough.md
[Getting Started]: ./getting-started.md

## Asking Questions

If you have questions, please make a post on the [Rust Zulip server][rust-zulip] or
[internals.rust-lang.org][internals]. If you are contributing to Rustup, be aware they are not on
Zulip - you can ask questions in `#wg-rustup` [on Discord][rust-discord].
See the [list of teams and working groups][governance] and [the Community page][community] on the
official website for more resources.

[governance]: https://www.rust-lang.org/governance
[community]: https://www.rust-lang.org/community

As a reminder, all contributors are expected to follow our [Code of Conduct][coc].

The compiler team (or `t-compiler`) usually hangs out in Zulip [in this
"stream"][z]; it will be easiest to get questions answered there.

[z]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler

**Please ask questions!** A lot of people report feeling that they are "wasting
expert time", but nobody on `t-compiler` feels this way. Contributors are
important to us.

Also, if you feel comfortable, prefer public topics, as this means others can
see the questions and answers, and perhaps even integrate them back into this
guide :)

### Experts

Not all `t-compiler` members are experts on all parts of `rustc`; it's a
pretty large project. To find out who could have some expertise on
different parts of the compiler, [consult triagebot assign groups][map].
The sections that start with `[assign*` in `triagebot.toml` file. 
But also, feel free to ask questions even if you can't figure out who to ping.

Another way to find experts for a given part of the compiler is to see who has made recent commits.
For example, to find people who have recently worked on name resolution since the 1.68.2 release,
you could run `git shortlog -n 1.68.2.. compiler/rustc_resolve/`. Ignore any commits starting with
"Rollup merge" or commits by `@bors` (see [CI contribution procedures](./contributing.md#ci) for
more information about these commits).

[map]: https://github.com/rust-lang/rust/blob/master/triagebot.toml

### Etiquette

We do ask that you be mindful to include as much useful information as you can
in your question, but we recognize this can be hard if you are unfamiliar with
contributing to Rust.

Just pinging someone without providing any context can be a bit annoying and
just create noise, so we ask that you be mindful of the fact that the
`t-compiler` folks get a lot of pings in a day.

## What should I work on?

The Rust project is quite large and it can be difficult to know which parts of the project need
help, or are a good starting place for beginners. Here are some suggested starting places.

### Easy or mentored issues

If you're looking for somewhere to start, check out the following [issue
search][help-wanted-search]. See the [Triage] for an explanation of these labels. You can also try
filtering the search to areas you're interested in. For example:

- `repo:rust-lang/rust-clippy` will only show clippy issues
- `label:T-compiler` will only show issues related to the compiler
- `label:A-diagnostics` will only show diagnostic issues

Not all important or beginner work has issue labels.
See below for how to find work that isn't labelled.

[help-wanted-search]: https://github.com/issues?q=is%3Aopen+is%3Aissue+org%3Arust-lang+no%3Aassignee+label%3AE-easy%2C%22good+first+issue%22%2Cgood-first-issue%2CE-medium%2CEasy%2CE-help-wanted%2CE-mentor+-label%3AS-blocked+-linked%3Apr+
[Triage]: ./contributing.md#issue-triage

### Recurring work

Some work is too large to be done by a single person. In this case, it's common to have "Tracking
issues" to co-ordinate the work between contributors. Here are some example tracking issues where
it's easy to pick up work without a large time commitment:

- [Move UI tests to subdirectories](https://github.com/rust-lang/rust/issues/73494)

If you find more recurring work, please feel free to add it here!

### Clippy issues

The [Clippy] project has spent a long time making its contribution process as friendly to newcomers
as possible. Consider working on it first to get familiar with the process and the compiler
internals.

See [the Clippy contribution guide][clippy-contributing] for instructions on getting started.

[Clippy]: https://doc.rust-lang.org/clippy/
[clippy-contributing]: https://github.com/rust-lang/rust-clippy/blob/master/CONTRIBUTING.md

### Diagnostic issues

Many diagnostic issues are self-contained and don't need detailed background knowledge of the
compiler. You can see a list of diagnostic issues [here][diagnostic-issues].

[diagnostic-issues]: https://github.com/rust-lang/rust/issues?q=is%3Aissue+is%3Aopen+label%3AA-diagnostics+no%3Aassignee

### Picking up abandoned pull requests

Sometimes, contributors send a pull request, but later find out that they don't have enough
time to work on it, or they simply are not interested in it anymore. Such PRs are often
eventually closed and they receive the `S-inactive` label. You could try to examine some of
these PRs and pick up the work. You can find the list of such PRs [here][abandoned-prs].

If the PR has been implemented in some other way in the meantime, the `S-inactive` label
should be removed from it. If not, and it seems that there is still interest in the change,
you can try to rebase the pull request on top of the latest `master` branch and send a new
pull request, continuing the work on the feature.

[abandoned-prs]: https://github.com/rust-lang/rust/pulls?q=is%3Apr+label%3AS-inactive+is%3Aclosed

### Writing tests

Issues that have been resolved but do not have a regression test are marked with the `E-needs-test` label. Writing unit tests is a low-risk, lower-priority task that offers new contributors a great opportunity to familiarize themselves with the testing infrastructure and contribution workflow.

### Contributing to std (standard library)

See [std-dev-guide](https://std-dev-guide.rust-lang.org/).

### Contributing code to other Rust projects

There are a bunch of other projects that you can contribute to outside of the
`rust-lang/rust` repo, including `cargo`, `miri`, `rustup`, and many others.

These repos might have their own contributing guidelines and procedures. Many
of them are owned by working groups. For more info, see the documentation in those repos' READMEs.

### Other ways to contribute

There are a bunch of other ways you can contribute, especially if you don't
feel comfortable jumping straight into the large `rust-lang/rust` codebase.

The following tasks are doable without much background knowledge but are
incredibly helpful:

- [Writing documentation][wd]: if you are feeling a bit more intrepid, you could try
  to read a part of the code and write doc comments for it. This will help you
  to learn some part of the compiler while also producing a useful artifact!
- [Triaging issues][triage]: categorizing, replicating, and minimizing issues is very helpful to the Rust maintainers.
- [Working groups][wg]: there are a bunch of working groups on a wide variety
  of rust-related things.
- Answer questions in the _Get Help!_ channels on the [Rust Discord
  server][rust-discord], on [users.rust-lang.org][users], or on
  [StackOverflow][so].
- Participate in the [RFC process](https://github.com/rust-lang/rfcs).
- Find a [requested community library][community-library], build it, and publish
  it to [Crates.io](http://crates.io). Easier said than done, but very, very
  valuable!

[rust-discord]: https://discord.gg/rust-lang
[users]: https://users.rust-lang.org/
[so]: http://stackoverflow.com/questions/tagged/rust
[community-library]: https://github.com/rust-lang/rfcs/labels/A-community-library
[wd]: ./contributing.md#writing-documentation
[wg]: https://rust-lang.github.io/compiler-team/working-groups/
[triage]: ./contributing.md#issue-triage

## Cloning and Building

See ["How to build and run the compiler"](./building/how-to-build-and-run.md).

## Contributor Procedures

This section has moved to the ["Contribution Procedures"](./contributing.md) chapter.

## Other Resources

This section has moved to the ["About this guide"][more-links] chapter.

[more-links]: ./about-this-guide.md#other-places-to-find-information
