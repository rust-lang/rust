# Contributing to Rust

Thank you for your interest in contributing to Rust! There are many ways to contribute
and we appreciate all of them.

The best way to get started is by asking for help in the [#new
members](https://rust-lang.zulipchat.com/#narrow/stream/122652-new-members)
Zulip stream. We have a lot of documentation below on how to get started on your own, but
the Zulip stream is the best place to *ask* for help.

Documentation for contributing to the compiler or tooling is located in the [Guide to Rustc
Development][rustc-dev-guide], commonly known as the [rustc-dev-guide]. Documentation for the
standard library in the [Standard library developers Guide][std-dev-guide], commonly known as the [std-dev-guide].

## Making changes to subtrees and submodules

For submodules, changes need to be made against the repository corresponding to the
submodule, and not the main `rust-lang/rust` repository.

For subtrees, prefer sending a PR against the subtree's repository if it does
not need to be made against the main `rust-lang/rust` repository (e.g. a
rustc-dev-guide change that does not accompany a compiler change).

## About the [rustc-dev-guide]

The [rustc-dev-guide] is meant to help document how rustc –the Rust compiler– works,
as well as to help new contributors get involved in rustc development. It is recommended
that you read and understand the [rustc-dev-guide] before making a contribution. This guide
talks about the different bots in the Rust ecosystem, the Rust development tools,
bootstrapping, the compiler architecture, source code representation, and more.

## [Getting help](https://rustc-dev-guide.rust-lang.org/getting-started.html#asking-questions)

There are many ways you can get help when you're stuck. Rust has many platforms for this:
[internals], [rust-zulip], and [rust-discord]. It is recommended to ask for help on
the [rust-zulip], but any of these platforms are great ways to seek help and even
find a mentor! You can learn more about asking questions and getting help in the
[Asking Questions](https://rustc-dev-guide.rust-lang.org/getting-started.html#asking-questions) chapter of the [rustc-dev-guide].

## Bug reports

Did a compiler error message tell you to come here? If you want to create an ICE report,
refer to [this section][contributing-bug-reports] and [open an issue][issue template].

[rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/
[std-dev-guide]: https://std-dev-guide.rust-lang.org/
[contributing-bug-reports]: https://rustc-dev-guide.rust-lang.org/contributing.html#bug-reports
[issue template]: https://github.com/rust-lang/rust/issues/new/choose
[internals]: https://internals.rust-lang.org
[rust-discord]: http://discord.gg/rust-lang
[rust-zulip]: https://rust-lang.zulipchat.com
