# Architecture

This document describes high-level architecture of rust-analyzer.
If you want to familiarize yourself with the code base, you are just
in the right place!


## Code generation

Some of the components of this repository are generated through automatic
processes. These are outlined below:

- `gen-kinds`: The kinds of tokens are reused in several places, so a generator
  is used. This process uses [tera] to generate, using data in [grammar.ron],
  the files:
  - [ast/generated.rs][ast generated] in `ra_syntax` based on
    [ast/generated.tera.rs][ast source]
  - [syntax_kinds/generated.rs][syntax_kinds generated] in `ra_syntax` based on
    [syntax_kinds/generated.tera.rs][syntax_kinds source]

[tera]: https://tera.netlify.com/
[grammar.ron]: ./crates/ra_syntax/src/grammar.ron
[ast generated]: ./crates/ra_syntax/src/ast/generated.rs
[ast source]: ./crates/ra_syntax/src/ast/generated.tera.rs
[syntax_kinds generated]: ./crates/ra_syntax/src/syntax_kinds/generated.rs
[syntax_kinds source]: ./crates/ra_syntax/src/syntax_kinds/generated.tera.rs


## Code Walk-Through

### `crates/ra_syntax`

Rust syntax tree structure and parser. See
[RFC](https://github.com/rust-lang/rfcs/pull/2256) for some design
notes.

- [rowan](https://github.com/rust-analyzer/rowan) library is used for constructing syntax trees.
- `grammar` module is the actual parser. It is a hand-written recursive descent parsers, which
  produced a sequence of events like "start node X", "finish not Y". It works similarly to  [kotlin parser](https://github.com/JetBrains/kotlin/blob/4d951de616b20feca92f3e9cc9679b2de9e65195/compiler/frontend/src/org/jetbrains/kotlin/parsing/KotlinParsing.java),
  which is a good source for inspiration for dealing with syntax errors and incomplete input. Original [libsyntax parser](https://github.com/rust-lang/rust/blob/6b99adeb11313197f409b4f7c4083c2ceca8a4fe/src/libsyntax/parse/parser.rs)
  is what we use for the definition of the Rust language.
- `parser_api/parser_impl` bridges the tree-agnostic parser from `grammar` with `rowan` trees.
  This is the thing that turns a flat list of events into a tree (see `EventProcessor`)
- `ast` a type safe API on top of the raw `rowan` tree.
- `grammar.ron` RON description of the grammar, which is used to
  generate `syntax_kinds` and `ast` modules, using `cargo gen-kinds` command.
- `algo`: generic tree algorithms, including `walk` for O(1) stack
  space tree traversal (this is cool) and `visit` for type-driven
  visiting the nodes (this is double plus cool, if you understand how
  `Visitor` works, you understand rust-analyzer).

Test for ra_syntax are mostly data-driven: `tests/data/parser` contains a bunch of `.rs`
(test vectors) and `.txt` files with corresponding syntax trees. During testing, we check
`.rs` against `.txt`. If the `.txt` file is missing, it is created (this is how you update
tests). Additionally, running `cargo gen-tests` will walk the grammar module and collect
all `//test test_name` comments into files inside `tests/data` directory.

See [#93](https://github.com/rust-analyzer/rust-analyzer/pull/93) for an example PR which
fixes a bug in the grammar.


### `crates/ra_editor`

All IDE features which can be implemented if you only have access to a
single file. `ra_editor` could be used to enhance editing of Rust code
without the need to fiddle with build-systems, file
synchronization and such.

In a sense, `ra_editor` is just a bunch of pure functions which take a
syntax tree as an input.

The tests for `ra_editor` are `[cfg(test)] mod tests` unit-tests spread
throughout its modules.

### `crates/salsa`

An implementation of red-green incremental compilation algorithm from
rust compiler. It makes all rust-analyzer features on-demand. To be replaced
with `salsa-rs/salsa` soon.


### `crates/ra_analysis`

A stateful library for analyzing many Rust files as they change.
`AnalysisHost` is a mutable entity (clojure's atom) which holds
current state, incorporates changes and handles out `Analysis` --- an
immutable consistent snapshot of world state at a point in time, which
actually powers analysis.


### `crates/ra_lsp_server`

An LSP implementation which uses `ra_analysis` for managing state and
`ra_editor` for actually doing useful stuff.

See [#79](https://github.com/rust-analyzer/rust-analyzer/pull/79/) as an
example of PR which adds a new feature to `ra_editor` and exposes it
to `ra_lsp_server`.


### `crates/cli`

A CLI interface to rust-analyzer.

### `crate/tools`

Code-gen tasks, used to develop rust-analyzer:

- `cargo gen-kinds` -- generate `ast` and `syntax_kinds`
- `cargo gen-tests` -- collect inline tests from grammar
- `cargo install-code` -- build and install VS Code extension and server

### `editors/code`

VS Code plugin


## Common workflows

To try out VS Code extensions, run `cargo install-code`. To see logs from the language server,
set `RUST_LOG=info` env variable. To see all communication between the server and the client, use
`RUST_LOG=gen_lsp_server=debug` (will print quite a bit of stuff).

To run tests, just `cargo test`.

To work on VS Code extension, launch code inside `editors/code` and use `F5` to launch/debug.

