# Appendix D: Code Index

rustc has a lot of important data structures. This is an attempt to give some
guidance on where to learn more about some of the key data structures of the
compiler.

Item            |  Kind    | Short description           | Chapter            | Declaration
----------------|----------|-----------------------------|--------------------|-------------------
`CodeMap` | struct | The CodeMap maps the AST nodes to their source code | [The parser](the-parser.html) | [src/libsyntax/codemap.rs](https://github.com/rust-lang/rust/blob/master/src/libsyntax/codemap.rs)
`CompileState` | struct | State that is passed to a callback at each compiler pass | [The Rustc Driver](rustc-driver.html) | [src/librustc_driver/driver.rs](https://github.com/rust-lang/rust/blob/master/src/librustc_driver/driver.rs)
`ast::Crate` | struct | Syntax-level representation of a parsed crate | | [src/librustc/hir/mod.rs](https://github.com/rust-lang/rust/blob/master/src/libsyntax/ast.rs)
`hir::Crate` | struct | Top-level data structure representing the crate being compiled | | [src/librustc/hir/mod.rs](https://github.com/rust-lang/rust/blob/master/src/librustc/hir/mod.rs)
`ParseSess` | struct | This struct contains information about a parsing session | [The parser](the-parser.html) | [src/libsyntax/parse/mod.rs](https://github.com/rust-lang/rust/blob/master/src/libsyntax/parse/mod.rs)
`StringReader` | struct | This is the lexer used during parsing. It consumes characters from the raw source code being compiled and produces a series of tokens for use by the rest of the parser | [The parser](the-parser.html) |  [src/libsyntax/parse/lexer/mod.rs](https://github.com/rust-lang/rust/blob/master/src/libsyntax/parse/lexer/mod.rs)
`Session` | struct | The data associated with a compilation session | [the Parser](the-parser.html), [The Rustc Driver](rustc-driver.html) | [src/librustc/session/mod.html](https://github.com/rust-lang/rust/blob/master/src/librustc/session/mod.rs)
`TraitDef` | struct | This struct contains a trait's definition with type information | [The `ty` modules](ty.html) |  [src/librustc/ty/trait_def.rs](https://github.com/rust-lang/rust/blob/master/src/librustc/ty/trait_def.rs)
`TyCtxt<'cx, 'tcx, 'tcx>` | type | The "typing context". This is the central data structure in the compiler. It is the context that you use to perform all manner of queries. | [The `ty` modules](ty.html) | [src/librustc/ty/context.rs](https://github.com/rust-lang/rust/blob/master/src/librustc/ty/context.rs)
