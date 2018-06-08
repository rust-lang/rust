# Appendix C: Glossary

The compiler uses a number of...idiosyncratic abbreviations and things. This
glossary attempts to list them and give you a few pointers for understanding
them better.

Term                    | Meaning
------------------------|--------
AST                     |  the abstract syntax tree produced by the syntax crate; reflects user syntax very closely.
binder                  |  a "binder" is a place where a variable or type is declared; for example, the `<T>` is a binder for the generic type parameter `T` in `fn foo<T>(..)`, and \|`a`\|` ...` is a binder for the parameter `a`. See [the background chapter for more](./appendix/background.html#free-vs-bound)
bound variable          |  a "bound variable" is one that is declared within an expression/term. For example, the variable `a` is bound within the closure expession \|`a`\|` a * 2`. See [the background chapter for more](./appendix/background.html#free-vs-bound)
codegen unit            |  when we produce LLVM IR, we group the Rust code into a number of codegen units. Each of these units is processed by LLVM independently from one another, enabling parallelism. They are also the unit of incremental re-use.
completeness            |  completeness is a technical term in type theory. Completeness means that every type-safe program also type-checks. Having both soundness and completeness is very hard, and usually soundness is more important. (see "soundness").
control-flow graph      |  a representation of the control-flow of a program; see [the background chapter for more](./appendix/background.html#cfg)
CTFE                    |  Compile-Time Function Evaluation. This is the ability of the compiler to evaluate `const fn`s at compile time. This is part of the compiler's constant evaluation system. ([see more](./const-eval.html))
cx                      |  we tend to use "cx" as an abbrevation for context. See also `tcx`, `infcx`, etc.
DAG                     |  a directed acyclic graph is used during compilation to keep track of dependencies between queries. ([see more](incremental-compilation.html))
data-flow analysis      |  a static analysis that figures out what properties are true at each point in the control-flow of a program; see [the background chapter for more](./appendix/background.html#dataflow)
DefId                   |  an index identifying a definition (see `librustc/hir/def_id.rs`). Uniquely identifies a `DefPath`.
Double pointer          |  a pointer with additional metadata. See "fat pointer" for more.
DST                     |  Dynamically-Sized Type. A type for which the compiler cannot statically know the size in memory (e.g. `str` or `[u8]`). Such types don't implement `Sized` and cannot be allocated on the stack. They can only occur as the last field in a struct. They can only be used behind a pointer (e.g. `&str` or `&[u8]`).
empty type              |  see "uninhabited type".
Fat pointer             |  a two word value carrying the address of some value, along with some further information necessary to put the value to use. Rust includes two kinds of "fat pointers": references to slices, and trait objects. A reference to a slice carries the starting address of the slice and its length. A trait object carries a value's address and a pointer to the trait's implementation appropriate to that value. "Fat pointers" are also known as "wide pointers", and "double pointers".
free variable           |  a "free variable" is one that is not bound within an expression or term; see [the background chapter for more](./appendix/background.html#free-vs-bound)
'gcx                    |  the lifetime of the global arena ([see more](ty.html))
generics                |  the set of generic type parameters defined on a type or item
HIR                     |  the High-level IR, created by lowering and desugaring the AST ([see more](hir.html))
HirId                   |  identifies a particular node in the HIR by combining a def-id with an "intra-definition offset".
HIR Map                 |  The HIR map, accessible via tcx.hir, allows you to quickly navigate the HIR and convert between various forms of identifiers.
ICE                     |  internal compiler error. When the compiler crashes.
ICH                     |  incremental compilation hash. ICHs are used as fingerprints for things such as HIR and crate metadata, to check if changes have been made. This is useful in incremental compilation to see if part of a crate has changed and should be recompiled.
inference variable      |  when doing type or region inference, an "inference variable" is a kind of special type/region that represents what you are trying to infer. Think of X in algebra. For example, if we are trying to infer the type of a variable in a program, we create an inference variable to represent that unknown type.
infcx                   |  the inference context (see `librustc/infer`)
IR                      |  Intermediate Representation. A general term in compilers. During compilation, the code is transformed from raw source (ASCII text) to various IRs. In Rust, these are primarily HIR, MIR, and LLVM IR. Each IR is well-suited for some set of computations. For example, MIR is well-suited for the borrow checker, and LLVM IR is well-suited for codegen because LLVM accepts it.
local crate             |  the crate currently being compiled.
LTO                     |  Link-Time Optimizations. A set of optimizations offered by LLVM that occur just before the final binary is linked. These include optmizations like removing functions that are never used in the final program, for example. _ThinLTO_ is a variant of LTO that aims to be a bit more scalable and efficient, but possibly sacrifices some optimizations. You may also read issues in the Rust repo about "FatLTO", which is the loving nickname given to non-Thin LTO. LLVM documentation: [here][lto] and [here][thinlto]
[LLVM]                  |  (actually not an acronym :P) an open-source compiler backend. It accepts LLVM IR and outputs native binaries. Various languages (e.g. Rust) can then implement a compiler front-end that output LLVM IR and use LLVM to compile to all the platforms LLVM supports.
MIR                     |  the Mid-level IR that is created after type-checking for use by borrowck and trans ([see more](./mir/index.html))
miri                    |  an interpreter for MIR used for constant evaluation ([see more](./miri.html))
normalize               |  a general term for converting to a more canonical form, but in the case of rustc typically refers to [associated type normalization](./traits/associated-types.html#normalize)
newtype                 |  a "newtype" is a wrapper around some other type (e.g., `struct Foo(T)` is a "newtype" for `T`). This is commonly used in Rust to give a stronger type for indices.
NLL                     | [non-lexical lifetimes](./mir/regionck.html), an extension to Rust's borrowing system to make it be based on the control-flow graph.
node-id or NodeId       |  an index identifying a particular node in the AST or HIR; gradually being phased out and replaced with `HirId`.
obligation              |  something that must be proven by the trait system ([see more](traits/resolution.html))
projection              |  a general term for a "relative path", e.g. `x.f` is a "field projection", and `T::Item` is an ["associated type projection"](./traits/goals-and-clauses.html#trait-ref)
promoted constants      |  constants extracted from a function and lifted to static scope; see [this section](./mir/index.html#promoted) for more details.
provider                |  the function that executes a query ([see more](query.html))
quantified              |  in math or logic, existential and universal quantification are used to ask questions like "is there any type T for which is true?" or "is this true for all types T?"; see [the background chapter for more](./appendix/background.html#quantified)
query                   |  perhaps some sub-computation during compilation ([see more](query.html))
region                  |  another term for "lifetime" often used in the literature and in the borrow checker.
rib                     |  a data structure in the name resolver that keeps track of a single scope for names. ([see more](./name-resolution.html))
sess                    |  the compiler session, which stores global data used throughout compilation
side tables             |  because the AST and HIR are immutable once created, we often carry extra information about them in the form of hashtables, indexed by the id of a particular node.
sigil                   |  like a keyword but composed entirely of non-alphanumeric tokens. For example, `&` is a sigil for references.
skolemization           |  a way of handling subtyping around "for-all" types (e.g., `for<'a> fn(&'a u32)`) as well as solving higher-ranked trait bounds (e.g., `for<'a> T: Trait<'a>`). See [the chapter on skolemization and universes](./mir/regionck.html#skol) for more details.
soundness               |  soundness is a technical term in type theory. Roughly, if a type system is sound, then if a program type-checks, it is type-safe; i.e. I can never (in safe rust) force a value into a variable of the wrong type. (see "completeness").
span                    |  a location in the user's source code, used for error reporting primarily. These are like a file-name/line-number/column tuple on steroids: they carry a start/end point, and also track macro expansions and compiler desugaring. All while being packed into a few bytes (really, it's an index into a table). See the Span datatype for more.
substs                  |  the substitutions for a given generic type or item (e.g. the `i32`, `u32` in `HashMap<i32, u32>`)
tcx                     |  the "typing context", main data structure of the compiler ([see more](ty.html))
'tcx                    |  the lifetime of the currently active inference context ([see more](ty.html))
trait reference         |  the name of a trait along with a suitable set of input type/lifetimes ([see more](./traits/goals-and-clauses.html#trait-ref))
token                   |  the smallest unit of parsing. Tokens are produced after lexing ([see more](the-parser.html)).
[TLS]                   |  Thread-Local Storage. Variables may be defined so that each thread has its own copy (rather than all threads sharing the variable). This has some interactions with LLVM. Not all platforms support TLS.
trans                   |  the code to translate MIR into LLVM IR.
trait reference         |  a trait and values for its type parameters ([see more](ty.html)).
ty                      |  the internal representation of a type ([see more](ty.html)).
UFCS                    |  Universal Function Call Syntax. An unambiguous syntax for calling a method ([see more](type-checking.html)).
uninhabited type        |  a type which has _no_ values. This is not the same as a ZST, which has exactly 1 value. An example of an uninhabited type is `enum Foo {}`, which has no variants, and so, can never be created. The compiler can treat code that deals with uninhabited types as dead code, since there is no such value to be manipulated. `!` (the never type) is an uninhabited type. Uninhabited types are also called "empty types".
upvar                   |  a variable captured by a closure from outside the closure.
variance                |  variance determines how changes to a generic type/lifetime parameter affect subtyping; for example, if `T` is a subtype of `U`, then `Vec<T>` is a subtype `Vec<U>` because `Vec` is *covariant* in its generic parameter. See [the background chapter](./appendix/background.html#variance) for a more general explanation. See the [variance chapter](./variance.html) for an explanation of how type checking handles variance.
Wide pointer            |  a pointer with additional metadata. See "fat pointer" for more.
ZST                     |  Zero-Sized Type. A type whose values have size 0 bytes. Since `2^0 = 1`, such types can have exactly one value. For example, `()` (unit) is a ZST. `struct Foo;` is also a ZST. The compiler can do some nice optimizations around ZSTs.

[LLVM]: https://llvm.org/
[lto]: https://llvm.org/docs/LinkTimeOptimization.html
[thinlto]: https://clang.llvm.org/docs/ThinLTO.html
[TLS]: https://llvm.org/docs/LangRef.html#thread-local-storage-models
