# The Parser

The parser is responsible for converting raw Rust source code into a structured
form which is easier for the compiler to work with, usually called an [*Abstract
Syntax Tree*][ast]. An AST mirrors the structure of a Rust program in memory,
using a `Span` to link a particular AST node back to its source text.

The bulk of the parser lives in the [libsyntax] crate.

Like most parsers, the parsing process is composed of two main steps,

- lexical analysis – turn a stream of characters into a stream of token trees
- parsing – turn the token trees into an AST

The `syntax` crate contains several main players,

- a [`CodeMap`] for mapping AST nodes to their source code
- the [ast module] contains types corresponding to each AST node
- a [`StringReader`] for lexing source code into tokens
- the [parser module] and [`Parser`] struct are in charge of actually parsing
  tokens into AST nodes,
- and a [visit module] for walking the AST and inspecting or mutating the AST
  nodes.

The main entrypoint to the parser is via the various `parse_*` functions
in the [parser module]. They let you do things like turn a filemap into a
token stream, create a parser from the token stream, and then execute the
parser to get a `Crate` (the root AST node).

To minimise the amount of copying that is done, both the `StringReader` and
`Parser` have lifetimes which bind them to the parent `ParseSess`. This contains
all the information needed while parsing, as well as the `CodeMap` itself.

[libsyntax]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/index.html
[rustc_errors]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html
[ast]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
[`CodeMap`]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/codemap/struct.CodeMap.html
[ast module]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast/index.html
[parser module]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/index.html
[`Parser`]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/parser/struct.Parser.html
[`StringReader`]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/lexer/struct.StringReader.html
[visit module]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/visit/index.html
