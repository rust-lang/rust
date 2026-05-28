# Lexing and parsing

The very first thing the compiler does is take the program (in UTF-8 Unicode text)
and turn it into a data format the compiler can work with more conveniently than strings.
This happens in two stages: Lexing and Parsing.

  1. _Lexing_ takes strings and turns them into streams of [tokens]. For
  example, `foo.bar + buz` would be turned into the tokens `foo`, `.`, `bar`,
  `+`, and `buz`. This is implemented in [`rustc_lexer`][lexer].

[tokens]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/token/index.html
[lexer]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lexer/index.html

  2. _Parsing_ takes streams of tokens and turns them into a structured form
  which is easier for the compiler to work with, usually called an [*Abstract
  Syntax Tree* (AST)][ast] . 

## The AST

The AST mirrors the structure of a Rust program in memory, using a `Span` to
link a particular AST node back to its source text. The AST is defined in
[`rustc_ast`][rustc_ast], along with some definitions for tokens and token
streams, data structures/traits for mutating ASTs, and shared definitions for
other AST-related parts of the compiler (like the lexer and
macro-expansion).

Every node in the AST has its own [`NodeId`], including top-level items
such as structs, but also individual statements and expressions. A [`NodeId`]
is an identifier number that uniquely identifies an AST node within a crate.

However, because they are absolute within a crate, adding or removing a single
node in the AST causes all the subsequent [`NodeId`]s to change. This renders
[`NodeId`]s pretty much useless for incremental compilation, where you want as
few things as possible to change.

[`NodeId`]s are used in all the `rustc` bits that operate directly on the AST,
like macro expansion and name resolution (more on these over the next couple chapters).

[`NodeId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/node_id/struct.NodeId.html

## Parsing

The parser is defined in [`rustc_parse`][rustc_parse], along with a
high-level interface to the lexer and some validation routines that run after
macro expansion. In particular, the [`rustc_parse::parser`][parser] contains
the parser implementation.

The main entrypoint to the parser is via the various `parse_*` functions and others in
[rustc_parse][rustc_parse]. They let you do things like turn a [`SourceFile`][sourcefile]
(e.g. the source in a single file) into a token stream, create a parser from
the token stream, and then execute the parser to get a [`Crate`] (the root AST
node).

To minimize the amount of copying that is done,
both [`Lexer`] and [`Parser`] have lifetimes which bind them to the parent [`ParseSess`].
This contains all the information needed while parsing, as well as the [`SourceMap`] itself.

Note that while parsing, we may encounter macro definitions or invocations.
We set these aside to be expanded (see [Macro Expansion](./macro-expansion.md)).
Expansion itself may require parsing the output of a macro, which may reveal more macros to be expanded, and so on.

## More on lexical analysis

Code for lexical analysis is split between two crates:

- [`rustc_lexer`] crate is responsible for breaking a `&str` into chunks
  constituting tokens. Although it is popular to implement lexers as generated
  finite state machines, the lexer in [`rustc_lexer`] is hand-written.

- [`Lexer`] integrates [`rustc_lexer`] with data structures specific to
  `rustc`. Specifically, it adds `Span` information to tokens returned by
  [`rustc_lexer`] and interns identifiers.

[`Crate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/struct.Crate.html
[`Parser`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/parser/struct.Parser.html
[`ParseSess`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/parse/struct.ParseSess.html
[`rustc_lexer`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lexer/index.html
[`SourceMap`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/source_map/struct.SourceMap.html
[`Lexer`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/lexer/struct.Lexer.html
[ast module]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/index.html
[ast]: ./ast-validation.md
[parser]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/parser/index.html
[rustc_ast]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/index.html
[rustc_errors]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html
[rustc_parse]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/index.html
[sourcefile]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.SourceFile.html
[visit module]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/visit/index.html
