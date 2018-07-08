# Appendix D: Code Index

rustc has a lot of important data structures. This is an attempt to give some
guidance on where to learn more about some of the key data structures of the
compiler.

Item            |  Kind    | Short description           | Chapter            | Declaration
----------------|----------|-----------------------------|--------------------|-------------------
`BodyId` | struct | One of four types of HIR node identifiers. | [Identifiers in the HIR] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/struct.BodyId.html)
`CodeMap` | struct | The CodeMap maps the AST nodes to their source code | [The parser] | [src/libsyntax/codemap.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/codemap/struct.CodeMap.html)
`CompileState` | struct | State that is passed to a callback at each compiler pass | [The Rustc Driver] | [src/librustc_driver/driver.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/driver/struct.CompileState.html)
`ast::Crate` | struct | Syntax-level representation of a parsed crate | [The parser] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast/struct.Crate.html)
`hir::Crate` | struct | More abstract, compiler-friendly form of a crate's AST | [The Hir] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/struct.Crate.html)
`DefId` | struct | One of four types of HIR node identifiers. | [Identifiers in the HIR] | [src/librustc/hir/def_id.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/def_id/struct.DefId.html)
`DiagnosticBuilder` | struct | A struct for building up compiler diagnostics, such as errors or lints | [Emitting Diagnostics] | [src/librustc_errors/diagnostic_builder.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagnosticBuilder.html)
`DocContext` | struct | A state container used by rustdoc when crawling through a crate to gather its documentation | [Rustdoc] | [src/librustdoc/core.rs](https://github.com/rust-lang/rust/blob/master/src/librustdoc/core.rs)
`FileMap` | struct | A single source within a `CodeMap` (e.g. the source code within a single file). | [The parser] | [src/libsyntax_pos/lib.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/codemap/struct.FileMap.html)
`HirId` | struct | One of four types of HIR node identifiers. | [Identifiers in the HIR] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/struct.HirId.html)
`NodeId` | struct | One of four types of HIR node identifiers. Being phased out. | [Identifiers in the HIR] | [src/libsyntax/ast.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast/struct.NodeId.html)
`ParamEnv` | struct | Information about generic parameters or `Self`, useful for working with associated or generic items | [Parameter Environment] | [src/librustc/ty/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.ParamEnv.html)
`ParseSess` | struct | This struct contains information about a parsing session | [The parser] | [src/libsyntax/parse/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/struct.ParseSess.html)
`Rib` | struct | Represents a single scope of names | [Name resolution] | [src/librustc_resolve/lib.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Rib.html)
`Session` | struct | The data associated with a compilation session | [The parser], [The Rustc Driver] | [src/librustc/session/mod.html](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/session/struct.Session.html)
`Span` | struct  | A location in the user's source code, used for error reporting primarily | [Emitting Diagnostics] | [src/libsyntax_pos/span_encoding.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax_pos/struct.Span.html)
`StringReader` | struct | This is the lexer used during parsing. It consumes characters from the raw source code being compiled and produces a series of tokens for use by the rest of the parser | [The parser] |  [src/libsyntax/parse/lexer/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/lexer/struct.StringReader.html)
`syntax::token_stream::TokenStream` | struct | An abstract sequence of tokens, organized into `TokenTree`s | [The parser], [Macro expansion] | [src/libsyntax/tokenstream.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/tokenstream/struct.TokenStream.html)
`TraitDef` | struct | This struct contains a trait's definition with type information | [The `ty` modules] |  [src/librustc/ty/trait_def.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/trait_def/struct.TraitDef.html)
`TraitRef` | struct | The combination of a trait and its input types (e.g. `P0: Trait<P1...Pn>`) | [Trait Solving: Goals and Clauses], [Trait Solving: Lowering impls]  |  [src/librustc/ty/sty.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TraitRef.html)
`Ty<'tcx>` | struct | This is the internal representation of a type used for type checking | [Type checking] | [src/librustc/ty/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/type.Ty.html)
`TyCtxt<'cx, 'tcx, 'tcx>` | type | The "typing context". This is the central data structure in the compiler. It is the context that you use to perform all manner of queries. | [The `ty` modules] | [src/librustc/ty/context.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TyCtxt.html)

[The HIR]: hir.html
[Identifiers in the HIR]: hir.html#hir-id
[The parser]: the-parser.html
[The Rustc Driver]: rustc-driver.html
[Type checking]: type-checking.html
[The `ty` modules]: ty.html
[Rustdoc]: rustdoc.html
[Emitting Diagnostics]: diag.html
[Macro expansion]: macro-expansion.html
[Name resolution]: name-resolution.html
[Parameter Environment]: param_env.html
[Trait Solving: Goals and Clauses]: traits/goals-and-clauses.html#domain-goals
[Trait Solving: Lowering impls]: traits/lowering-rules.html#lowering-impls
