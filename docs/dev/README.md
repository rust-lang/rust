# Contributing Quick Start

Rust Analyzer is just a usual rust project, which is organized as a Cargo
workspace, builds on stable and doesn't depend on C libraries. So, just

```
$ cargo test
```

should be enough to get you started!

To learn more about how rust-analyzer works, see
[./architecture.md](./architecture.md) document.

We also publish rustdoc docs to pages:

https://rust-analyzer.github.io/rust-analyzer/ra_ide_api/index.html

Various organizational and process issues are discussed in this document.

# Getting in Touch

Rust Analyzer is a part of [RLS-2.0 working
group](https://github.com/rust-lang/compiler-team/tree/6a769c13656c0a6959ebc09e7b1f7c09b86fb9c0/working-groups/rls-2.0).
Discussion happens in this Zulip stream:

https://rust-lang.zulipchat.com/#narrow/stream/185405-t-compiler.2Fwg-rls-2.2E0

# Work List

We have this "work list" paper document:

https://paper.dropbox.com/doc/RLS-2.0-work-list--AZ3BgHKKCtqszbsi3gi6sjchAQ-42vbnxzuKq2lKwW0mkn8Y

It shows what everyone is working on right now. If you want to (this is not
mandatory), add yourself to the list!

# Issue Labels

* [good-first-issue](https://github.com/rust-analyzer/rust-analyzer/labels/good%20first%20issue)
  are good issues to get into the project.
* [E-mentor](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-mentor)
  issues have links to the code in question and tests.
* [E-easy](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-easy),
  [E-medium](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-medium),
  [E-hard](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-hard),
  labels are *estimates* for how hard would be to write a fix.
* [fun](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3Afun)
  is for cool, but probably hard stuff.

# CI

We use Travis for CI. Most of the things, including formatting, are checked by
`cargo test` so, if `cargo test` passes locally, that's a good sign that CI will
be green as well. We use bors-ng to enforce the [not rocket
science](https://graydon2.dreamwidth.org/1597.html) rule.

You can run `cargo format-hook` to install git-hook to run rustfmt on commit.

# Code organization

All Rust code lives in the `crates` top-level directory, and is organized as a
single Cargo workspace. The `editors` top-level directory contains code for
integrating with editors. Currently, it contains plugins for VS Code (in
typescript) and Emacs (in elisp). The `docs` top-level directory contains both
developer and user documentation.

We have some automation infra in Rust in the `crates/tool` package. It contains
stuff like formatting checking, code generation and powers `cargo install-ra`.
The latter syntax is achieved with the help of cargo aliases (see `.cargo`
directory).

# Launching rust-analyzer

Debugging language server can be tricky: LSP is rather chatty, so driving it
from the command line is not really feasible, driving it via VS Code requires
interacting with two processes.

For this reason, the best way to see how rust-analyzer works is to find a
relevant test and execute it (VS Code includes an action for running a single
test).

However, launching a VS Code instance with locally build language server is
possible. There's even a VS Code task for this, so just <kbd>F5</kbd> should
work (thanks, [@andrew-w-ross](https://github.com/andrew-w-ross)!).

I often just install development version with `cargo install-ra --server --jemalloc` and
restart the host VS Code.

See [./debugging.md](./debugging.md) for how to attach to rust-analyzer with
debugger, and don't forget that rust-analyzer has useful `pd` snippet and `dbg`
postfix completion for printf debugging :-)

# Working With VS Code Extension

To work on the VS Code extension, launch code inside `editors/code` and use `F5`
to launch/debug. To automatically apply formatter and linter suggestions, use
`npm run fix`.

Tests are located inside `src/test` and are named `*.test.ts`. They use the
[Mocha](https://mochajs.org) test framework and the builtin Node
[assert](https://nodejs.org/api/assert.html) module. Unlike normal Node tests
they must be hosted inside a VS Code instance. This can be done in one of two
ways:

1. When `F5` debugging in VS Code select the `Extension Tests` configuration
   from the drop-down at the top of the Debug View. This will launch a temporary
   instance of VS Code. The test results will appear in the "Debug Console" tab
   of the primary VS Code instance.

2. Run `npm test` from the command line. Although this is initiated from the
   command line it is not headless; it will also launch a temporary instance of
   VS Code.

Due to the requirements of running the tests inside VS Code they are **not run
on CI**. When making changes to the extension please ensure the tests are not
broken locally before opening a Pull Request.

# Logging

Logging is done by both rust-analyzer and VS Code, so it might be tricky to
figure out where logs go.

Inside rust-analyzer, we use the standard `log` crate for logging, and
`flexi_logger` for logging frotend. By default, log goes to stderr (the same as
with `env_logger`), but the stderr itself is processed by VS Code. To mirror
logs to a `./log` directory, set `RA_LOG_DIR=1` environmental variable.

To see stderr in the running VS Code instance, go to the "Output" tab of the
panel and select `rust-analyzer`. This shows `eprintln!` as well. Note that
`stdout` is used for the actual protocol, so `println!` will break things.

To log all communication between the server and the client, there are two choices:

* you can log on the server side, by running something like
  ```
  env RUST_LOG=gen_lsp_server=trace code .
  ```

* you can log on the client side, by enabling `"rust-analyzer.trace.server":
  "verbose"` workspace setting. These logs are shown in a separate tab in the
  output and could be used with LSP inspector. Kudos to
  [@DJMcNab](https://github.com/DJMcNab) for setting this awesome infra up!


There's also two VS Code commands which might be of interest:

* `Rust Analyzer: Status` shows some memory-usage statistics. To take full
  advantage of it, you need to compile rust-analyzer with jemalloc support:
  ```
  $ cargo install --path crates/ra_lsp_server --force --features jemalloc
  ```

  There's an alias for this: `cargo install-ra --server --jemalloc`.

* `Rust Analyzer: Syntax Tree` shows syntax tree of the current file/selection.

# Profiling

We have a built-in hierarchical profiler, you can enable it by using `RA_PROF` env-var:

```
RA_PROFILE=*             // dump everything
RA_PROFILE=foo|bar|baz   // enabled only selected entries
RA_PROFILE=*@3>10        // dump everything, up to depth 3, if it takes more than 10 ms
```

In particular, I have `export RA_PROFILE='*>10' in my shell profile.

To measure time for from-scratch analysis, use something like this:

```
$ cargo run --release -p ra_cli -- analysis-stats ../chalk/
```

For measuring time of incremental analysis, use either of these:

```
$ cargo run --release -p ra_cli -- analysis-bench ../chalk/ --highlight ../chalk/chalk-engine/src/logic.rs
$ cargo run --release -p ra_cli -- analysis-bench ../chalk/ --complete ../chalk/chalk-engine/src/logic.rs:94:0
```
