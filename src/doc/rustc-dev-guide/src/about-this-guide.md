# About this guide

This guide is meant to help document how rustc – the Rust compiler –
works, as well as to help new contributors get involved in rustc
development.

There are six parts to this guide:

1. [Building, Debugging, and Contributing to `rustc`][p1]: Contains information that should be useful no matter how
   you are contributing, such as procedures for contribution, building the
   compiler, etc.
2. [High-Level Compiler Architecture][p2]: Discusses the high-level
   architecture of the compiler and stages of the compile process.
3. [Source Code Representation][p3]: Describes the process of taking raw source code from the user and
   transforming it into various forms that the compiler can work with easily.
4. [Analysis][p4]: discusses the analyses that the compiler uses to check various
   properties of the code and inform later stages of the compile process (e.g., type checking).
5. [From MIR to Binaries][p5]: How linked executable machine code is generated.
6. [Appendices][app] at the end with useful reference information. There are a
   few of these with different information, inluding a glossary.

[p1]: ./part-1-intro.md
[p2]: ./part-2-intro.md
[p3]: ./part-3-intro.md
[p4]: ./part-4-intro.md
[p5]: ./part-5-intro.md
[app]: ./appendix/background.md

The Guide itself is of course open-source as well, and the sources can
be found at the [GitHub repository]. If you find any mistakes in the
guide, please file an issue about it, or even better, open a PR
with a correction!

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
