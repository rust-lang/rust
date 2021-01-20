# About this guide

This guide is meant to help document how rustc – the Rust compiler –
works, as well as to help new contributors get involved in rustc
development.

There are six parts to this guide:

1. [Building and debugging `rustc`][p1]: Contains information that should be
   useful no matter how you are contributing, about building, debugging,
   profiling, etc.
2. [Contributing to `rustc`][p1-5]: Contains information that should be useful
   no matter how you are contributing, about procedures for contribution,
   stabilizing features, etc.
2. [High-Level Compiler Architecture][p2]: Discusses the high-level
   architecture of the compiler and stages of the compile process.
3. [Source Code Representation][p3]: Describes the process of taking raw source code from the user and
   transforming it into various forms that the compiler can work with easily.
4. [Analysis][p4]: discusses the analyses that the compiler uses to check various
   properties of the code and inform later stages of the compile process (e.g., type checking).
5. [From MIR to Binaries][p5]: How linked executable machine code is generated.
6. [Appendices][app] at the end with useful reference information. There are a
   few of these with different information, including a glossary.

[p1]: ./getting-started.md
[p1-5]: ./compiler-team.md
[p2]: ./part-2-intro.md
[p3]: ./part-3-intro.md
[p4]: ./part-4-intro.md
[p5]: ./part-5-intro.md
[app]: ./appendix/background.md

### Constant change

Keep in mind that `rustc` is a real production-quality product, being worked upon continuously by a
sizeable set of contributors.
As such, it has its fair share of codebase churn and technical debt.
In addition, many of the ideas discussed throughout this guide are idealized designs that are not
fully realized yet.
All this makes keeping this guide completely up to date on everything very hard!

The Guide itself is of course open-source as well, and the sources can be found at the
[GitHub repository].
If you find any mistakes in the guide, please file an issue about it, or even better, open a PR with
a correction!

If you do contribute to the guide, please see the corresponding
[subsection on writing documentation in this guide].

[subsection on writing documentation in this guide]: contributing.md#contributing-to-rustc-dev-guide

> “‘All conditioned things are impermanent’ — when one sees this with wisdom, one turns away from
> suffering.” _The Dhammapada, verse 277_

## Other places to find information

You might also find the following sites useful:

- [rustc API docs] -- rustdoc documentation for the compiler
- [Forge] -- contains documentation about rust infrastructure, team procedures, and more
- [compiler-team] -- the home-base for the rust compiler team, with description
  of the team procedures, active working groups, and the team calendar.

[GitHub repository]: https://github.com/rust-lang/rustc-dev-guide/
[rustc API docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/
[Forge]: https://forge.rust-lang.org/
[compiler-team]: https://github.com/rust-lang/compiler-team/
