The `rustc_ast` crate contains those things concerned purely with syntax
– that is, the AST ("abstract syntax tree"), along with some definitions for tokens and token streams, data structures/traits for mutating ASTs, and shared definitions for other AST-related parts of the compiler (like the lexer and macro-expansion).

For more information about how these things work in rustc, see the
rustc dev guide:

- [Parsing](https://rustc-dev-guide.rust-lang.org/the-parser.html)
- [Macro Expansion](https://rustc-dev-guide.rust-lang.org/macro-expansion.html)
