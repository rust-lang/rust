# Code Index

rustc has a lot of important data structures. This is an attempt to give some
guidance on where to learn more about some of the key data structures of the
compiler.

Item            |  Kind    | Short description           | Chapter            | Declaration
----------------|----------|-----------------------------|--------------------|-------------------
`BodyId` | struct | One of four types of HIR node identifiers | [Identifiers in the HIR] | [compiler/rustc_hir/src/hir.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.BodyId.html)
`Compiler` | struct | Represents a compiler session and can be used to drive a compilation. | [The Rustc Driver and Interface] | [compiler/rustc_interface/src/interface.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/interface/struct.Compiler.html)
`ast::Crate` | struct | A syntax-level representation of a parsed crate | [The parser] | [compiler/rustc_ast/src/ast.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/struct.Crate.html)
`rustc_hir::Crate` | struct | A more abstract, compiler-friendly form of a crate's AST | [The Hir] | [compiler/rustc_hir/src/hir.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Crate.html)
`DefId` | struct | One of four types of HIR node identifiers | [Identifiers in the HIR] | [compiler/rustc_hir/src/def_id.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html)
`Diag` | struct | A struct for a compiler diagnostic, such as an error or lint | [Emitting Diagnostics] | [compiler/rustc_errors/src/diagnostic.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.Diag.html)
`DocContext` | struct | A state container used by rustdoc when crawling through a crate to gather its documentation | [Rustdoc] | [src/librustdoc/core.rs](https://github.com/rust-lang/rust/blob/master/src/librustdoc/core.rs)
`HirId` | struct | One of four types of HIR node identifiers | [Identifiers in the HIR] | [compiler/rustc_hir/src/hir_id.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir_id/struct.HirId.html)
`Lexer` | struct | This is the lexer used during parsing. It consumes characters from the raw source code being compiled and produces a series of tokens for use by the rest of the parser | [The parser] |  [compiler/rustc_parse/src/lexer/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_parse/lexer/struct.Lexer.html)
`NodeId` | struct | One of four types of HIR node identifiers. Being phased out | [Identifiers in the HIR] | [compiler/rustc_ast/src/ast.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/node_id/struct.NodeId.html)
`P` | struct | An owned immutable smart pointer. By contrast, `&T` is not owned, and `Box<T>` is not immutable. | None | [compiler/rustc_ast/src/ptr.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ptr/struct.P.html)
`ParamEnv` | struct | Information about generic parameters or `Self`, useful for working with associated or generic items | [Parameter Environment] | [compiler/rustc_middle/src/ty/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html)
`ParseSess` | struct | This struct contains information about a parsing session | [The parser] | [compiler/rustc_session/src/parse/parse.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/parse/struct.ParseSess.html)
`Rib` | struct | Represents a single scope of names | [Name resolution] | [compiler/rustc_resolve/src/lib.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.Rib.html)
`Session` | struct | The data associated with a compilation session | [The parser], [The Rustc Driver and Interface] | [compiler/rustc_session/src/session.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/struct.Session.html)
`SourceFile` | struct | Part of the `SourceMap`. Maps AST nodes to their source code for a single source file. Was previously called FileMap | [The parser] | [compiler/rustc_span/src/lib.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.SourceFile.html)
`SourceMap` | struct | Maps AST nodes to their source code. It is composed of `SourceFile`s. Was previously called CodeMap | [The parser] | [compiler/rustc_span/src/source_map.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/source_map/struct.SourceMap.html)
`Span` | struct  | A location in the user's source code, used for error reporting primarily | [Emitting Diagnostics] | [compiler/rustc_span/src/span_encoding.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.Span.html)
`rustc_ast::token_stream::TokenStream` | struct | An abstract sequence of tokens, organized into `TokenTree`s | [The parser], [Macro expansion] | [compiler/rustc_ast/src/tokenstream.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/tokenstream/struct.TokenStream.html)
`TraitDef` | struct | This struct contains a trait's definition with type information | [The `ty` modules] |  [compiler/rustc_middle/src/ty/trait_def.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/trait_def/struct.TraitDef.html)
`TraitRef` | struct | The combination of a trait and its input types (e.g. `P0: Trait<P1...Pn>`) | [Trait Solving: Goals and Clauses]  |  [compiler/rustc_middle/src/ty/sty.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.TraitRef.html)
`Ty<'tcx>` | struct | This is the internal representation of a type used for type checking | [Type checking] | [compiler/rustc_middle/src/ty/mod.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html)
`TyCtxt<'tcx>` | struct | The "typing context". This is the central data structure in the compiler. It is the context that you use to perform all manner of queries | [The `ty` modules] | [compiler/rustc_middle/src/ty/context.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html)

[The HIR]: ../hir.html
[Identifiers in the HIR]: ../hir.html#hir-id
[The parser]: ../the-parser.html
[The Rustc Driver and Interface]: ../rustc-driver/intro.html
[Type checking]: ../type-checking.html
[The `ty` modules]: ../ty.html
[Rustdoc]: ../rustdoc.html
[Emitting Diagnostics]: ../diagnostics.html
[Macro expansion]: ../macro-expansion.html
[Name resolution]: ../name-resolution.html
[Parameter Environment]: ../typing_parameter_envs.html
[Trait Solving: Goals and Clauses]: ../traits/goals-and-clauses.html#domain-goals
