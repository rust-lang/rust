# Clippy Development

Hello fellow Rustacean! If you made it here, you're probably interested in
making Clippy better by contributing to it. In that case, welcome to the
project!

> _Note:_ If you're just interested in using Clippy, there's nothing to see from
> this point onward, and you should return to one of the earlier chapters.

## Getting started

If this is your first time contributing to Clippy, you should first read the
[Basics docs](basics.md). This will explain the basics on how to get the source
code and how to compile and test the code.

## Additional Readings for Beginners

If a dear reader of this documentation has never taken a class on compilers
and interpreters, it might be confusing as to why AST level deals with only
the language's syntax. And some readers might not even understand what lexing,
parsing, and AST mean.

This documentation serves by no means as a crash course on compilers or language design.
And for details specifically related to Rust, the [Rustc Development Guide][rustc_dev_guide]
is a far better choice to peruse.

The [Syntax and AST][ast] chapter and the [High-Level IR][hir] chapter are
great introduction to the concepts mentioned in this chapter.

Some readers might also find the [introductory chapter][map_of_territory] of
Robert Nystrom's _Crafting Interpreters_ a helpful overview of compiled and
interpreted languages before jumping back to the Rustc guide.

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

[ast]: https://rustc-dev-guide.rust-lang.org/syntax-intro.html
[hir]: https://rustc-dev-guide.rust-lang.org/hir.html
[rustc_dev_guide]: https://rustc-dev-guide.rust-lang.org/
[map_of_territory]: https://craftinginterpreters.com/a-map-of-the-territory.html
[clippy_rfc]: https://github.com/rust-lang/rfcs/blob/master/text/2476-clippy-uno.md
[rfc_stability]: https://github.com/rust-lang/rfcs/blob/master/text/2476-clippy-uno.md#stability-guarantees
[rfc_lint_cats]: https://github.com/rust-lang/rfcs/blob/master/text/2476-clippy-uno.md#lint-audit-and-categories
[`CONTRIBUTING.md`]: https://github.com/rust-lang/rust-clippy/blob/master/CONTRIBUTING.md
