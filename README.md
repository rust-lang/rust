# libsyntax2.0

[![Build Status](https://travis-ci.org/matklad/libsyntax2.svg?branch=master)](https://travis-ci.org/matklad/libsyntax2)
[![Build status](https://ci.appveyor.com/api/projects/status/j56x1hbje8rdg6xk/branch/master?svg=true)](https://ci.appveyor.com/project/matklad/libsyntax2/branch/master)


libsyntax2.0 is an **experimental** parser of the Rust language,
intended for the use in IDEs.
[RFC](https://github.com/rust-lang/rfcs/pull/2256).


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

* **Go to symbol in workspace** (no support for Cargo deps yet)

* code actions:
  - Flip `,` in comma separated lists
  - Add `#[derive]` to struct/enum
  - Add `impl` block to struct/enum
  - Run tests at caret
  
* **Go to definition** ("correct" for `mod foo;` decls, index-based for functions).


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
