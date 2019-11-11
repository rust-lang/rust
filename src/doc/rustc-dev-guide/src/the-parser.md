# Lexing and Parsing

> The parser and lexer are currently undergoing a lot of refactoring, so parts
> of this chapter may be out of date.

The very first thing the compiler does is take the program (in Unicode
characters) and turn it into something the compiler can work with more
conveniently than strings. This happens in two stages: Lexing and Parsing.

Lexing takes strings and turns them into streams of tokens. For example,
`a.b + c` would be turned into the tokens `a`, `.`, `b`, `+`, and `c`.
The lexer lives in [`librustc_lexer`][lexer].

[lexer]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lexer/index.html

Parsing then takes streams of tokens and turns them into a structured
form which is easier for the compiler to work with, usually called an [*Abstract
Syntax Tree*][ast] (AST). An AST mirrors the structure of a Rust program in memory,
using a `Span` to link a particular AST node back to its source text.

The AST is defined in [`libsyntax`][libsyntax], along with some definitions for
tokens and token streams, data structures/traits for mutating ASTs, and shared
definitions for other AST-related parts of the compiler (like the lexer and
macro-expansion).

The parser is defined in [`librustc_parse`][librustc_parse], along with a
high-level interface to the lexer and some validation routines that run after
macro expansion. In particular, the [`rustc_parser::parser`][parser] contains
the parser implementation.

The main entrypoint to the parser is via the various `parse_*` functions in the
[parser][parser]. They let you do things like turn a [`SourceFile`][sourcefile]
(e.g. the source in a single file) into a token stream, create a parser from
the token stream, and then execute the parser to get a `Crate` (the root AST
node).

To minimise the amount of copying that is done, both the `StringReader` and
`Parser` have lifetimes which bind them to the parent `ParseSess`. This contains
all the information needed while parsing, as well as the `SourceMap` itself.

## More on Lexical Analysis

Code for lexical analysis is split between two crates:

- `rustc_lexer` crate is responsible for breaking a `&str` into chunks
  constituting tokens. Although it is popular to implement lexers as generated
  finite state machines, the lexer in `rustc_lexer` is hand-written.

- [`StringReader`] from [libsyntax] integrates `rustc_lexer` with `rustc`
  specific data structures. Specifically, it adds `Span` information to tokens
  returned by `rustc_lexer` and interns identifiers.

[libsyntax]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/index.html
[rustc_errors]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html
[ast]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
[`SourceMap`]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/source_map/struct.SourceMap.html
[ast module]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast/index.html
[librustc_parse]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/index.html
[parser]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/parser/index.html
[`Parser`]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/parser/struct.Parser.html
[`StringReader`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/lexer/struct.StringReader.html
[visit module]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/visit/index.html
[sourcefile]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/source_map/struct.SourceFile.html
