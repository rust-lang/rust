# Appendix D: Code Index

rustc has a lot of important data structures. This is an attempt to give some
guidance on where to learn more about some of the key data structures of the
compiler.

Item            |  Kind    | Short description           | Chapter            | Declaration
----------------|----------|-----------------------------|--------------------|-------------------
`BodyId` | struct | One of four types of HIR node identifiers | [Identifiers in the HIR] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/struct.BodyId.html)
`Compiler` | struct | Represents a compiler session and can be used to drive a compilation. | [The Rustc Driver and Interface] | [src/librustc_interface/interface.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Compiler.html)
`ast::Crate` | struct | A syntax-level representation of a parsed crate | [The parser] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast/struct.Crate.html)
`hir::Crate` | struct | A more abstract, compiler-friendly form of a crate's AST | [The Hir] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/struct.Crate.html)
`DefId` | struct | One of four types of HIR node identifiers | [Identifiers in the HIR] | [src/librustc/hir/def_id.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/def_id/struct.DefId.html)
`DiagnosticBuilder` | struct | A struct for building up compiler diagnostics, such as errors or lints | [Emitting Diagnostics] | [src/librustc_errors/diagnostic_builder.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagnosticBuilder.html)
`DocContext` | struct | A state container used by rustdoc when crawling through a crate to gather its documentation | [Rustdoc] | [src/librustdoc/core.rs](https://github.com/rust-lang/rust/blob/master/src/librustdoc/core.rs)
`HirId` | struct | One of four types of HIR node identifiers | [Identifiers in the HIR] | [src/librustc/hir/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/hir/struct.HirId.html)
`NodeId` | struct | One of four types of HIR node identifiers. Being phased out | [Identifiers in the HIR] | [src/libsyntax/ast.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast/struct.NodeId.html)
`P` | struct | An owned immutable smart pointer. By contrast, `&T` is not owned, and `Box<T>` is not immutable. | None | [src/syntax/ptr.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ptr/struct.P.html)
`ParamEnv` | struct | Information about generic parameters or `Self`, useful for working with associated or generic items | [Parameter Environment] | [src/librustc/ty/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.ParamEnv.html)
`ParseSess` | struct | This struct contains information about a parsing session | [The parser] | [src/libsyntax/parse/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/struct.ParseSess.html)
`Query` | struct | Represents the result of query to the `Compiler` interface and allows stealing, borrowing, and returning the results of compiler passes. | [The Rustc Driver and Interface] | [src/librustc_interface/queries.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/queries/struct.Query.html)
`Rib` | struct | Represents a single scope of names | [Name resolution] | [src/librustc_resolve/lib.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Rib.html)
`Session` | struct | The data associated with a compilation session | [The parser], [The Rustc Driver and Interface] | [src/librustc/session/mod.html](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/session/struct.Session.html)
`SourceFile` | struct | Part of the `SourceMap`. Maps AST nodes to their source code for a single source file. Was previously called FileMap | [The parser] | [src/libsyntax_pos/lib.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/source_map/struct.SourceFile.html)
`SourceMap` | struct | Maps AST nodes to their source code. It is composed of `SourceFile`s. Was previously called CodeMap | [The parser] | [src/libsyntax/source_map.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/source_map/struct.SourceMap.html)
`Span` | struct  | A location in the user's source code, used for error reporting primarily | [Emitting Diagnostics] | [src/libsyntax_pos/span_encoding.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax_pos/struct.Span.html)
`StringReader` | struct | This is the lexer used during parsing. It consumes characters from the raw source code being compiled and produces a series of tokens for use by the rest of the parser | [The parser] |  [src/libsyntax/parse/lexer/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/parse/lexer/struct.StringReader.html)
`syntax::token_stream::TokenStream` | struct | An abstract sequence of tokens, organized into `TokenTree`s | [The parser], [Macro expansion] | [src/libsyntax/tokenstream.rs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/tokenstream/struct.TokenStream.html)
`TraitDef` | struct | This struct contains a trait's definition with type information | [The `ty` modules] |  [src/librustc/ty/trait_def.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/trait_def/struct.TraitDef.html)
`TraitRef` | struct | The combination of a trait and its input types (e.g. `P0: Trait<P1...Pn>`) | [Trait Solving: Goals and Clauses], [Trait Solving: Lowering impls]  |  [src/librustc/ty/sty.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TraitRef.html)
`Ty<'tcx>` | struct | This is the internal representation of a type used for type checking | [Type checking] | [src/librustc/ty/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/type.Ty.html)
`TyCtxt<'cx, 'tcx, 'tcx>` | struct | The "typing context". This is the central data structure in the compiler. It is the context that you use to perform all manner of queries | [The `ty` modules] | [src/librustc/ty/context.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TyCtxt.html)

[The HIR]: ../hir.html
[Identifiers in the HIR]: ../hir.html#hir-id
[The parser]: ../the-parser.html
[The Rustc Driver and Interface]: ../rustc-driver.html
[Type checking]: ../type-checking.html
[The `ty` modules]: ../ty.html
[Rustdoc]: ../rustdoc.html
[Emitting Diagnostics]: ../diag.html
[Macro expansion]: ../macro-expansion.html
[Name resolution]: ../name-resolution.html
[Parameter Environment]: ../param_env.html
[Trait Solving: Goals and Clauses]: ../traits/goals-and-clauses.html#domain-goals
[Trait Solving: Lowering impls]: ../traits/lowering-rules.html#lowering-impls
