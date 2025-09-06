# About this guide

This guide is meant to help document how rustc – the Rust compiler – works,
as well as to help new contributors get involved in rustc development.

There are several parts to this guide:

1. [Building and debugging `rustc`][p1]:
   Contains information that should be useful no matter how you are contributing,
   about building, debugging, profiling, etc.
1. [Contributing to Rust][p2]:
   Contains information that should be useful no matter how you are contributing,
   about procedures for contribution, using git and Github, stabilizing features, etc.
1. [Bootstrapping][p3]:
   Describes how the Rust compiler builds itself using previous versions, including
   an introduction to the bootstrap process and debugging methods.
1. [High-level Compiler Architecture][p4]:
   Discusses the high-level architecture of the compiler and stages of the compile process.
1. [Source Code Representation][p5]:
   Describes the process of taking raw source code from the user
   and transforming it into various forms that the compiler can work with easily.
1. [Supporting Infrastructure][p6]:
   Covers command-line argument conventions, compiler entry points like rustc_driver and
   rustc_interface, and the design and implementation of errors and lints.
1. [Analysis][p7]:
   Discusses the analyses that the compiler uses to check various properties of the code
   and inform later stages of the compile process (e.g., type checking).
1. [MIR to Binaries][p8]: How linked executable machine code is generated.
1. [Appendices][p9] at the end with useful reference information.
   There are a few of these with different information, including a glossary.

[p1]: ./building/how-to-build-and-run.html
[p2]: ./contributing.md
[p3]: ./building/bootstrapping/intro.md
[p4]: ./part-2-intro.md
[p5]: ./part-3-intro.md
[p6]: ./cli.md
[p7]: ./part-4-intro.md
[p8]: ./part-5-intro.md
[p9]: ./appendix/background.md

### Constant change

Keep in mind that `rustc` is a real production-quality product,
being worked upon continuously by a sizeable set of contributors.
As such, it has its fair share of codebase churn and technical debt.
In addition, many of the ideas discussed throughout this guide are idealized designs
that are not fully realized yet.
All this makes keeping this guide completely up to date on everything very hard!

The Guide itself is of course open-source as well,
and the sources can be found at the [GitHub repository].
If you find any mistakes in the guide, please file an issue about it.
Even better, open a PR with a correction!

If you do contribute to the guide,
please see the corresponding [subsection on writing documentation in this guide].

[subsection on writing documentation in this guide]: contributing.md#contributing-to-rustc-dev-guide

> “‘All conditioned things are impermanent’ — 
> when one sees this with wisdom, one turns away from suffering.”
> _The Dhammapada, verse 277_

## Other places to find information

You might also find the following sites useful:

- This guide contains information about how various parts of the
  compiler work and how to contribute to the compiler.
- [rustc API docs] -- rustdoc documentation for the compiler, devtools, and internal tools
- [Forge] -- contains documentation about Rust infrastructure, team procedures, and more
- [compiler-team] -- the home-base for the Rust compiler team, with description
  of the team procedures, active working groups, and the team calendar.
- [std-dev-guide] -- a similar guide for developing the standard library.
- [The t-compiler Zulip][z]
- The [Rust Internals forum][rif], a place to ask questions and
  discuss Rust's internals
- The [Rust reference][rr], even though it doesn't specifically talk about
  Rust's internals, is a great resource nonetheless
- Although out of date, [Tom Lee's great blog article][tlgba] is very helpful
- The [Rust Compiler Testing Docs][rctd]
- For [@bors], [this cheat sheet][cheatsheet] is helpful
- Google is always helpful when programming.
  You can [search all Rust documentation][gsearchdocs] (the standard library,
  the compiler, the books, the references, and the guides) to quickly find
  information about the language and compiler.
- You can also use Rustdoc's built-in search feature to find documentation on
  types and functions within the crates you're looking at. You can also search
  by type signature! For example, searching for `* -> vec` should find all
  functions that return a `Vec<T>`.
  _Hint:_ Find more tips and keyboard shortcuts by typing `?` on any Rustdoc
  page!


[rustc dev guide]: about-this-guide.md
[gsearchdocs]: https://www.google.com/search?q=site:doc.rust-lang.org+your+query+here
[stddocs]: https://doc.rust-lang.org/std
[rif]: http://internals.rust-lang.org
[rr]: https://doc.rust-lang.org/book/
[rustforge]: https://forge.rust-lang.org/
[tlgba]: https://tomlee.co/2014/04/a-more-detailed-tour-of-the-rust-compiler/
[ro]: https://www.rustaceans.org/
[rctd]: tests/intro.md
[cheatsheet]: https://bors.rust-lang.org/
[Miri]: https://github.com/rust-lang/miri
[@bors]: https://github.com/bors
[GitHub repository]: https://github.com/rust-lang/rustc-dev-guide/
[rustc API docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle
[Forge]: https://forge.rust-lang.org/
[compiler-team]: https://github.com/rust-lang/compiler-team/
[std-dev-guide]: https://std-dev-guide.rust-lang.org/
[z]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler
