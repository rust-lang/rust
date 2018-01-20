# The Parser

The parser is responsible for converting raw Rust source code into a structured
form which is easier for the compiler to work with, usually called an *Abstract
Syntax Tree*. The bulk of the parser lives in the [libsyntax] crate.

The parsing process is made up of roughly 3 stages,

- lexical analysis - turn a stream of characters into a stream of token trees
- macro expansion - run `proc-macros` and expand `macro_rules` macros
- parsing - turn the token trees into an AST


[libsyntax]: https://github.com/rust-lang/rust/tree/master/src/libsyntax
