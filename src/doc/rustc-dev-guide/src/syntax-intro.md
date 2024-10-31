# Syntax and the AST

Working directly with source code is very inconvenient and error-prone.
Thus, before we do anything else, we convert raw source code into an
[Abstract Syntax Tree (AST)][AST]. It turns out that doing this involves a lot of work,
including [lexing, parsing], [macro expansion], [name resolution], conditional
compilation, [feature-gate checking], and [validation] of the [AST].
In this chapter, we take a look at all of these steps.

Notably, there isn't always a clean ordering between these tasks.
For example, macro expansion relies on name resolution to resolve the names of macros and imports.
And parsing requires macro expansion, which in turn may require parsing the output of the macro.

[AST]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/index.html
[macro expansion]: ./macro-expansion.md
[feature-gate checking]: ./feature-gate-ck.md
[lexing, parsing]: ./the-parser.md
[name resolution]: ./name-resolution.md
[validation]: ./ast-validation.md
