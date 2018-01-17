Glossary
--------

The compiler uses a number of...idiosyncratic abbreviations and things. This glossary attempts to list them and give you a few pointers for understanding them better.

Term                    | Meaning
------------------------|--------
AST                     |  the abstract syntax tree produced by the syntax crate; reflects user syntax very closely.
codegen unit            |  when we produce LLVM IR, we group the Rust code into a number of codegen units. Each of these units is processed by LLVM independently from one another, enabling parallelism. They are also the unit of incremental re-use.
cx                      |  we tend to use "cx" as an abbrevation for context. See also `tcx`, `infcx`, etc.
DefId                   |  an index identifying a definition (see `librustc/hir/def_id.rs`). Uniquely identifies a `DefPath`.
HIR                     |  the High-level IR, created by lowering and desugaring the AST. See `librustc/hir`.
HirId                   |  identifies a particular node in the HIR by combining a def-id with an "intra-definition offset".
'gcx                    |  the lifetime of the global arena (see `librustc/ty`).
generics                |  the set of generic type parameters defined on a type or item
ICE                     |  internal compiler error. When the compiler crashes.
infcx                   |  the inference context (see `librustc/infer`)
MIR                     |  the Mid-level IR that is created after type-checking for use by borrowck and trans. Defined in the `src/librustc/mir/` module, but much of the code that manipulates it is found in `src/librustc_mir`.
obligation              |  something that must be proven by the trait system; see `librustc/traits`.
local crate             |  the crate currently being compiled.
node-id or NodeId       |  an index identifying a particular node in the AST or HIR; gradually being phased out and replaced with `HirId`.
query                   |  perhaps some sub-computation during compilation; see `librustc/maps`.
provider                |  the function that executes a query; see `librustc/maps`.
sess                    |  the compiler session, which stores global data used throughout compilation
side tables             |  because the AST and HIR are immutable once created, we often carry extra information about them in the form of hashtables, indexed by the id of a particular node.
span                    |  a location in the user's source code, used for error reporting primarily. These are like a file-name/line-number/column tuple on steroids: they carry a start/end point, and also track macro expansions and compiler desugaring. All while being packed into a few bytes (really, it's an index into a table). See the Span datatype for more.
substs                  |  the substitutions for a given generic type or item (e.g., the `i32`, `u32` in `HashMap<i32, u32>`)
tcx                     |  the "typing context", main data structure of the compiler (see `librustc/ty`).
trans                   |  the code to translate MIR into LLVM IR.
trait reference         |  a trait and values for its type parameters (see `librustc/ty`).
ty                      |  the internal representation of a type (see `librustc/ty`).
