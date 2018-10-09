# Rust Analyzer

[![Build Status](https://travis-ci.org/rust-analyzer/rust-analyzer.svg?branch=master)](https://travis-ci.org/rust-analyzer/rust-analyzer)
[![Build status](https://ci.appveyor.com/api/projects/status/vtt455oi3hjy9uvk/branch/master?svg=true)](https://ci.appveyor.com/project/matklad/rust-analyzer/branch/master)


Rust Analyzer is an **experimental** modular compiler frontend for the
Rust language, which aims to lay a foundation for excellent IDE
support.

It doesn't implement much of compiler functionality yet, but the
white-space preserving Rust parser works, and there are significant
chunks of overall architecture (indexing, on-demand & lazy
computation, snapshotable world view) in place. Some basic IDE
functionality is provided via a language server.

## Quick Start

Rust analyzer builds on stable Rust >= 1.29.0.

```
# run tests
$ cargo test

# show syntax tree of a Rust file
$ cargo run --package ra_cli parse < crates/ra_syntax/src/lib.rs

# show symbols of a Rust file
$ cargo run --package ra_cli symbols < crates/ra_syntax/src/lib.rs
```

To try out the language server, see [these
instructions](./editors/README.md). Please note that the server is not
ready for general use yet. If you are looking for a Rust IDE that
works, use [IntelliJ
Rust](https://github.com/intellij-rust/intellij-rust) or
[RLS](https://github.com/rust-lang-nursery/rls). That being said, the
basic stuff works, and rust analyzer is developed in the rust analyzer
powered editor.


## Current Status and Plans

Rust analyzer aims to fill the same niche as the official [Rust
Language Server](https://github.com/rust-lang-nursery/rls), but uses a
significantly different architecture. More details can be found [in
this
thread](https://internals.rust-lang.org/t/2019-strategy-for-rustc-and-the-rls/8361),
but the core issue is that RLS works in the "wait until user stops
typing, run the build process, save the results of the analysis" mode,
which arguably is the wrong foundation for IDE.

Rust Analyzer is a hobby project at the moment, there's exactly zero
guarantees that it becomes production-ready one day.

The near/mid term plan is to work independently of the main rustc
compiler and implement at least simplistic versions of name
resolution, macro expansion and type inference. The purpose is two
fold:

* to quickly bootstrap usable and useful language server: solution
  that covers 80% of Rust code will be useful for IDEs, and will be
  vastly simpler than 100% solution.

* to understand how the consumer-side of compiler API should look like
  (especially it's on-demand aspects). If you have
  `get_expression_type` function, you can write a ton of purely-IDE
  features on top of it, even if the function is only partially
  correct. Plugin in the precise function afterwards should just make
  IDE features more reliable.

The long term plan is to merge with the mainline rustc compiler,
probably around the HIR boundary? That is, use rust analyzer for
parsing, macro expansion and related bits of name resolution, but
leave the rest (including type inference and trait selection) to the
existing rustc.

## Getting in touch

@matklad can be found at Rust
[discord](https://discordapp.com/invite/rust-lang), in
#ides-and-editors.


## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) and [ARCHITECTURE.md](./ARCHITECTURE.md)


## License

Rust analyzer is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0).

See LICENSE-APACHE and LICENSE-MIT for details.
