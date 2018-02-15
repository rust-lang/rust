Glossary
--------

The compiler uses a number of...idiosyncratic abbreviations and things. This glossary attempts to list them and give you a few pointers for understanding them better.

Term                    | Meaning
------------------------|--------
AST                     |  the abstract syntax tree produced by the syntax crate; reflects user syntax very closely.
codegen unit            |  when we produce LLVM IR, we group the Rust code into a number of codegen units. Each of these units is processed by LLVM independently from one another, enabling parallelism. They are also the unit of incremental re-use.
completeness            |  completeness is a technical term in type theory. Completeness means that every type-safe program also type-checks. Having both soundness and completeness is very hard, and usually soundness is more important. (see "soundness").
cx                      |  we tend to use "cx" as an abbrevation for context. See also `tcx`, `infcx`, etc.
DAG                     |  a directed acyclic graph is used during compilation to keep track of dependencies between queries. ([see more](incremental-compilation.html))
DefId                   |  an index identifying a definition (see `librustc/hir/def_id.rs`). Uniquely identifies a `DefPath`.
HIR                     |  the High-level IR, created by lowering and desugaring the AST ([see more](hir.html))
HirId                   |  identifies a particular node in the HIR by combining a def-id with an "intra-definition offset".
'gcx                    |  the lifetime of the global arena ([see more](ty.html))
generics                |  the set of generic type parameters defined on a type or item
ICE                     |  internal compiler error. When the compiler crashes.
ICH                     |  incremental compilation hash. ICHs are used as fingerprints for things such as HIR and crate metadata, to check if changes have been made. This is useful in incremental compilation to see if part of a crate has changed and should be recompiled.
infcx                   |  the inference context (see `librustc/infer`)
local crate             |  the crate currently being compiled.
MIR                     |  the Mid-level IR that is created after type-checking for use by borrowck and trans ([see more](./mir.html))
node-id or NodeId       |  an index identifying a particular node in the AST or HIR; gradually being phased out and replaced with `HirId`.
obligation              |  something that must be proven by the trait system ([see more](trait-resolution.html))
provider                |  the function that executes a query ([see more](query.html))
query                   |  perhaps some sub-computation during compilation ([see more](query.html))
region                  |  another term for "lifetime" often used in the literature and in the borrow checker.
sess                    |  the compiler session, which stores global data used throughout compilation
side tables             |  because the AST and HIR are immutable once created, we often carry extra information about them in the form of hashtables, indexed by the id of a particular node.
sigil                   |  like a keyword but composed entirely of non-alphanumeric tokens. For example, `&` is a sigil for references.
soundness               |  soundness is a technical term in type theory. Roughly, if a type system is sound, then if a program type-checks, it is type-safe; i.e. I can never (in safe rust) force a value into a variable of the wrong type. (see "completeness").
span                    |  a location in the user's source code, used for error reporting primarily. These are like a file-name/line-number/column tuple on steroids: they carry a start/end point, and also track macro expansions and compiler desugaring. All while being packed into a few bytes (really, it's an index into a table). See the Span datatype for more.
substs                  |  the substitutions for a given generic type or item (e.g. the `i32`, `u32` in `HashMap<i32, u32>`)
tcx                     |  the "typing context", main data structure of the compiler ([see more](ty.html))
'tcx                    |  the lifetime of the currently active inference context ([see more](ty.html))
token                   |  the smallest unit of parsing. Tokens are produced after lexing ([see more](the-parser.html)).
trans                   |  the code to translate MIR into LLVM IR.
trait reference         |  a trait and values for its type parameters ([see more](ty.html)).
ty                      |  the internal representation of a type ([see more](ty.html)).
