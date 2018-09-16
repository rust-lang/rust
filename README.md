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

```
$ cargo test
$ cargo parse < crates/libsyntax2/src/lib.rs
```

## Trying It Out

This installs experimental VS Code plugin

```
$ cargo install-code
```

It's better to remove existing Rust plugins to avoid interference.
Warning: plugin is not intended for general use, has a lot of rough
edges and missing features (notably, no code completion). That said,
while originally libsyntax2 was developed in IntelliJ, @matklad now
uses this plugin (and thus, libsytax2) to develop libsyntax2, and it
doesn't hurt too much :-)


### Features:

* syntax highlighting (LSP does not have API for it, so impl is hacky
  and sometimes fall-backs to the horrible built-in highlighting)

* commands (`ctrl+shift+p` or keybindings)
  - **Show Rust Syntax Tree** (use it to verify that plugin works)
  - **Rust Extend Selection** (works with multiple cursors)
  - **Rust Matching Brace** (knows the difference between `<` and `<`)
  - **Rust Parent Module**
  - **Rust Join Lines** (deals with trailing commas)

* **Go to symbol in file**

* **Go to symbol in workspace**
  - `#Foo` searches for `Foo` type in the current workspace
  - `#foo#` searches for `foo` function in the current workspace
  - `#Foo*` searches for `Foo` type among dependencies, excluding `stdlib`
  - Sorry for a weired UI, neither LSP, not VSCode have any sane API for filtering! :)

* code actions:
  - Flip `,` in comma separated lists
  - Add `#[derive]` to struct/enum
  - Add `impl` block to struct/enum
  - Run tests at caret

* **Go to definition** ("correct" for `mod foo;` decls, index-based for functions).

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

### `crates/libsyntax2`

- `yellow`, red/green syntax tree, heavily inspired [by this](https://github.com/apple/swift/tree/ab68f0d4cbf99cdfa672f8ffe18e433fddc8b371/lib/Syntax)
- `grammar`, the actual parser
- `parser_api/parser_impl` bridges the tree-agnostic parser from `grammar` with `yellow` trees
- `grammar.ron` RON description of the grammar, which is used to
  generate `syntax_kinds` and `ast` modules.
- `algo`: generic tree algorithms, including `walk` for O(1) stack
  space tree traversal (this is cool) and `visit` for type-driven
  visiting the nodes (this is double plus cool, if you understand how
  `Visitor` works, you understand libsyntax2).


### `crates/libeditor`

Most of IDE features leave here, unlike `libanalysis`, `libeditor` is
single-file and is basically a bunch of pure functions.


### `crates/libanalysis`

A stateful library for analyzing many Rust files as they change.
`WorldState` is a mutable entity (clojure's atom) which holds current
state, incorporates changes and handles out `World`s --- immutable
consistent snapshots of `WorldState`, which actually power analysis.


### `crates/server`

An LSP implementation which uses `libanalysis` for managing state and
`libeditor` for actually doing useful stuff.


### `crates/cli`

A CLI interface to libsyntax

### `crate/tools`

Code-gen tasks, used to develop libsyntax2:

- `cargo gen-kinds` -- generate `ast` and `syntax_kinds`
- `cargo gen-tests` -- collect inline tests from grammar
- `cargo install-code` -- build and install VS Code extension and server

### `code`

VS Code plugin


## Performance

Non-incremental, but seems pretty fast:

```
$ cargo build --release --package cli
$ wc -l ~/projects/rust/src/libsyntax/parse/parser.rs
7546 /home/matklad/projects/rust/src/libsyntax/parse/parser.rs
$ ./target/release/cli parse < ~/projects/rust/src/libsyntax/parse/parser.rs --no-dump  > /dev/null
parsing: 21.067065ms
```

## Getting in touch

@matklad can be found at Rust
[discord](https://discordapp.com/invite/rust-lang), in
#ides-and-editors.


## License

libsyntax2 is primarily distributed under the terms of both the MIT license
and the Apache License (Version 2.0).

See LICENSE-APACHE and LICENSE-MIT for details.
