# Contributing Quick Start

Rust Analyzer is just a usual rust project, which is organized as a Cargo
workspace, builds on stable and doesn't depend on C libraries. So, just

```
$ cargo test
```

should be enough to get you started!

To learn more about how rust-analyzer works, see
[./architecture.md](./architecture.md) document.

Various organizational and process issues are discussed here.

# Getting in Touch

Rust Analyzer is a part of [RLS-2.0 working
group](https://github.com/rust-lang/compiler-team/tree/6a769c13656c0a6959ebc09e7b1f7c09b86fb9c0/working-groups/rls-2.0).
Discussion happens in this Zulip stream:

https://rust-lang.zulipchat.com/#narrow/stream/185405-t-compiler.2Fwg-rls-2.2E0

# Issue Labels

* [good-first-issue](https://github.com/rust-analyzer/rust-analyzer/labels/good%20first%20issue)
  are good issues to get into the project.
* [E-mentor](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-mentor)
  issues have links to the code in question and tests.
* [E-easy](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-easy),
  [E-medium](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-medium),
  [E-hard](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-hard),
  labels are *estimates* for how hard would be to write a fix.
* [E-fun](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-fun)
  is for cool, but probably hard stuff.

# CI

We use Travis for CI. Most of the things, including formatting, are checked by
`cargo test` so, if `cargo test` passes locally, that's a good sign that CI will
be green as well. We use bors-ng to enforce the [not rocket
science](https://graydon2.dreamwidth.org/1597.html) rule.
