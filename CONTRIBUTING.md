The project is in its early stages: contributions are welcome and would be
**very** helpful, but the project is not _yet_ optimized for contribution.
Moreover, it is doubly experimental, so there's no guarantee that any work here
would reach production. That said, here are some areas where contributions would
be **especially** welcome:

- Designing internal data structures: RFC only outlines the constraints, it's an
  open question how to satisfy them in the optimal way. See `ARCHITECTURE.md`
  for current design questions.

- Porting libsyntax parser to rust-analyzer: currently rust-analyzer parses only
  a tiny subset of Rust. This should be fixed by porting parsing functions from
  libsyntax one by one. Take a look at the [libsyntax parser] for "what to port"
  and at the [Kotlin parser] for "how to port".

- Writing validators: by design, rust-analyzer is very lax about the input. For
  example, the lexer happily accepts unclosed strings. The idea is that there
  should be a higher level visitor, which walks the syntax tree after parsing
  and produces all the warnings. Alas, there's no such visitor yet :( Would you
  like to write one? :)

- Creating tests: it would be tremendously helpful to read each of libsyntax and
  rust-analyzer parser functions and crate a small separate test cases to cover
  each and every edge case.

- Building stuff with rust-analyzer: it would be really cool to compile
  rust-analyzer to WASM and add _client side_ syntax validation to rust
  playground!

Do take a look at the issue tracker.

If you don't know where to start, or have _any_ questions or suggestions, don't
hesitate to chat at [Gitter]!

# Code generation

Some of the components of this repository are generated through automatic
processes. These are outlined below:

- `gen-kinds`: The kinds of tokens are reused in several places, so a generator
  is used. This process uses [tera] to generate, using data in [grammar.ron],
  the files:
  - [ast/generated.rs][ast generated] in `ra_syntax` based on
    [ast/generated.tera.rs][ast source]
  - [syntax_kinds/generated.rs][syntax_kinds generated] in `ra_syntax` based on
    [syntax_kinds/generated.tera.rs][syntax_kinds source]

[libsyntax parser]:
  https://github.com/rust-lang/rust/blob/6b99adeb11313197f409b4f7c4083c2ceca8a4fe/src/libsyntax/parse/parser.rs
[kotlin parser]:
  https://github.com/JetBrains/kotlin/blob/4d951de616b20feca92f3e9cc9679b2de9e65195/compiler/frontend/src/org/jetbrains/kotlin/parsing/KotlinParsing.java
[gitter]: https://gitter.im/libsyntax2/Lobby
[tera]: https://tera.netlify.com/
[grammar.ron]: ./crates/ra_syntax/src/grammar.ron
[ast generated]: ./crates/ra_syntax/src/ast/generated.rs
[ast source]: ./crates/ra_syntax/src/ast/generated.tera.rs
[syntax_kinds generated]: ./crates/ra_syntax/src/syntax_kinds/generated.rs
[syntax_kinds source]: ./crates/ra_syntax/src/syntax_kinds/generated.tera.rs
