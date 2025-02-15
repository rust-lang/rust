# Contributing to rust-analyzer

Thank you for your interest in contributing to rust-analyzer! There are many ways to contribute
and we appreciate all of them.

To get a quick overview of the crates and structure of the project take a look at the
[Contributing](https://rust-analyzer.github.io/book/contributing) section of the manual.

If you have any questions please ask them in the [rust-analyzer zulip stream](
https://rust-lang.zulipchat.com/#narrow/stream/185405-t-compiler.2Frust-analyzer) or if unsure where
to start out when working on a concrete issue drop a comment on the related issue for mentoring
instructions (general discussions are recommended to happen on zulip though).

## Fixing a bug or improving a feature

Generally it's fine to just work on these kinds of things and put a pull-request out for it. If there
is an issue accompanying it make sure to link it in the pull request description so it can be closed
afterwards or linked for context.

If you want to find something to fix or work on keep a look out for the `C-bug` and `C-enhancement`
labels.

## Implementing a new feature

It's advised to first open an issue for any kind of new feature so the team can tell upfront whether
the feature is desirable or not before any implementation work happens. We want to minimize the
possibility of someone putting a lot of work into a feature that is then going to waste as we deem
it out of scope (be it due to generally not fitting in with rust-analyzer, or just not having the
maintenance capacity). If there already is a feature issue open but it is not clear whether it is
considered accepted feel free to just drop a comment and ask!
