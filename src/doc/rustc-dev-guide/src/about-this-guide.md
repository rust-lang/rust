# About this guide

This guide is meant to help document how rustc – the Rust compiler – works,
as well as to help new contributors get involved in rustc development.

There are seven parts to this guide:

1. [Building and debugging `rustc`][p1]:
   Contains information that should be useful no matter how you are contributing,
   about building, debugging, profiling, etc.
2. [Contributing to `rustc`][p2]:
   Contains information that should be useful no matter how you are contributing,
   about procedures for contribution, using git and Github, stabilizing features, etc.
3. [High-Level Compiler Architecture][p3]:
   Discusses the high-level architecture of the compiler and stages of the compile process.
4. [Source Code Representation][p4]:
   Describes the process of taking raw source code from the user
   and transforming it into various forms that the compiler can work with easily.
5. [Analysis][p5]:
   discusses the analyses that the compiler uses to check various properties of the code
   and inform later stages of the compile process (e.g., type checking).
6. [From MIR to Binaries][p6]: How linked executable machine code is generated.
7. [Appendices][p7] at the end with useful reference information.
   There are a few of these with different information, including a glossary.

[p1]: ./getting-started.md
[p2]: ./contributing.md
[p3]: ./part-2-intro.md
[p4]: ./part-3-intro.md
[p5]: ./part-4-intro.md
[p6]: ./part-5-intro.md
[p7]: ./appendix/background.md

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

- [rustc API docs] -- rustdoc documentation for the compiler
- [Forge] -- contains documentation about Rust infrastructure, team procedures, and more
- [compiler-team] -- the home-base for the Rust compiler team, with description
  of the team procedures, active working groups, and the team calendar.
- [std-dev-guide] -- a similar guide for developing the standard library.

[GitHub repository]: https://github.com/rust-lang/rustc-dev-guide/
[rustc API docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/
[Forge]: https://forge.rust-lang.org/
[compiler-team]: https://github.com/rust-lang/compiler-team/
[std-dev-guide]: https://std-dev-guide.rust-lang.org/
