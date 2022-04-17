# Clippy Development

Hello fellow Rustacean! If you made it here, you're probably interested in
making Clippy better by contributing to it. In that case, welcome to the
project!

> _Note:_ If you're just interested in using Clippy, there's nothing to see from
> this point onward and you should return to one of the earlier chapters.

## Getting started

If this is your first time contributing to Clippy, you should first read the
[Basics docs](basics.md). This will explain the basics on how to get the source
code and how to compile and test the code.

## Writing code

If you have done the basic setup, it's time to start hacking.

The [Adding lints](adding_lints.md) chapter is a walk through on how to add a
new lint to Clippy. This is also interesting if you just want to fix a lint,
because it also covers how to test lints and gives an overview of the bigger
picture.

If you want to add a new lint or change existing ones apart from bugfixing, it's
also a good idea to give the [stability guarantees][rfc_stability] and
[lint categories][rfc_lint_cats] sections of the [Clippy 1.0 RFC][clippy_rfc] a
quick read. The lint categories are also described [earlier in this
book](../lints.md).

> _Note:_ Some higher level things about contributing to Clippy are still
> covered in the [`CONTRIBUTING.md`] document. Some of those will be moved to
> the book over time, like:
> - Finding something to fix
> - IDE setup
> - High level overview on how Clippy works
> - Triage procedure
> - Bors and Homu

[clippy_rfc]: https://github.com/rust-lang/rfcs/blob/master/text/2476-clippy-uno.md
[rfc_stability]: https://github.com/rust-lang/rfcs/blob/master/text/2476-clippy-uno.md#stability-guarantees
[rfc_lint_cats]: https://github.com/rust-lang/rfcs/blob/master/text/2476-clippy-uno.md#lint-audit-and-categories
[`CONTRIBUTING.md`]: https://github.com/rust-lang/rust-clippy/blob/master/CONTRIBUTING.md
