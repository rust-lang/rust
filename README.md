# Rust Analyzer

[![Build Status](https://travis-ci.org/matklad/rust-analyzer.svg?branch=master)](https://travis-ci.org/matklad/rust-analyzer)
[![Build status](https://ci.appveyor.com/api/projects/status/j56x1hbje8rdg6xk/branch/master?svg=true)](https://ci.appveyor.com/project/matklad/rust-analyzer/branch/master)


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
Language Server](https://github.com/rust-lang-nursery/rls), but
better. It was created because @matklad is not satisfied with RLS
original starting point and current direction. More details can be
found [in this
thread](https://internals.rust-lang.org/t/2019-strategy-for-rustc-and-the-rls/8361).
The core issue is that RLS works in the "wait until user stops typing,
run the build process, save the results of the analysis" mode, which
arguably is the wrong foundation for IDE (see the thread for details).

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

## Code Walk-Through

### `crates/ra_syntax`

Rust syntax tree structure and parser. See
[RFC](https://github.com/rust-lang/rfcs/pull/2256) for some design
notes.

- `yellow`, red/green syntax tree, heavily inspired [by this](https://github.com/apple/swift/tree/ab68f0d4cbf99cdfa672f8ffe18e433fddc8b371/lib/Syntax)
- `grammar`, the actual parser
- `parser_api/parser_impl` bridges the tree-agnostic parser from `grammar` with `yellow` trees
- `grammar.ron` RON description of the grammar, which is used to
  generate `syntax_kinds` and `ast` modules.
- `algo`: generic tree algorithms, including `walk` for O(1) stack
  space tree traversal (this is cool) and `visit` for type-driven
  visiting the nodes (this is double plus cool, if you understand how
  `Visitor` works, you understand rust-analyzer).


### `crates/ra_editor`

All IDE features which can be implemented if you only have access to a
single file. `ra_editor` could be used to enhance editing of Rust code
without the need to fiddle with build-systems, file
synchronization and such.

In a sense, `ra_editor` is just a bunch of pure functions which take a
syntax tree as an input.

### `crates/salsa`

An implementation of red-green incremental compilation algorithm from
rust compiler. It makes all rust-analyzer features on-demand.


### `crates/ra_analysis`

A stateful library for analyzing many Rust files as they change.
`AnalysisHost` is a mutable entity (clojure's atom) which holds
current state, incorporates changes and handles out `Analysis` --- an
immutable consistent snapshot of world state at a point in time, which
actually powers analysis.


### `crates/ra_lsp_server`

An LSP implementation which uses `ra_analysis` for managing state and
`ra_editor` for actually doing useful stuff.


### `crates/cli`

A CLI interface to libsyntax

### `crate/tools`

Code-gen tasks, used to develop rust-analyzer:

- `cargo gen-kinds` -- generate `ast` and `syntax_kinds`
- `cargo gen-tests` -- collect inline tests from grammar
- `cargo install-code` -- build and install VS Code extension and server

### `editors/code`

VS Code plugin


## Performance

Non-incremental, but seems pretty fast:

```
$ cargo build --release --package ra_cli
$ wc -l ~/projects/rust/src/libsyntax/parse/parser.rs
7546 /home/matklad/projects/rust/src/libsyntax/parse/parser.rs
$ ./target/release/ra_cli parse < ~/projects/rust/src/libsyntax/parse/parser.rs --no-dump  > /dev/null
parsing: 21.067065ms
```

## Getting in touch

@matklad can be found at Rust
[discord](https://discordapp.com/invite/rust-lang), in
#ides-and-editors.


## License

Rust analyzer is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0).

See LICENSE-APACHE and LICENSE-MIT for details.
