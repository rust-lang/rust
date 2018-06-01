# High-level overview of the compiler source

## Crate structure

The main Rust repository consists of a `src` directory, under which
there live many crates. These crates contain the sources for the
standard library and the compiler.  This document, of course, focuses
on the latter.

Rustc consists of a number of crates, including `syntax`,
`rustc`, `rustc_back`, `rustc_codegen`, `rustc_driver`, and
many more. The source for each crate can be found in a directory
like `src/libXXX`, where `XXX` is the crate name.

(N.B. The names and divisions of these crates are not set in
stone and may change over time. For the time being, we tend towards a
finer-grained division to help with compilation time, though as incremental
compilation improves, that may change.)

The dependency structure of these crates is roughly a diamond:

```text
                  rustc_driver
                /      |       \
              /        |         \
            /          |           \
          /            v             \
rustc_codegen  rustc_borrowck   ...  rustc_metadata
          \            |            /
            \          |          /
              \        |        /
                \      v      /
                    rustc
                       |
                       v
                    syntax
                    /    \
                  /       \
           syntax_pos  syntax_ext
```

The `rustc_driver` crate, at the top of this lattice, is effectively
the "main" function for the rust compiler. It doesn't have much "real
code", but instead ties together all of the code defined in the other
crates and defines the overall flow of execution. (As we transition
more and more to the [query model], however, the
"flow" of compilation is becoming less centrally defined.)

At the other extreme, the `rustc` crate defines the common and
pervasive data structures that all the rest of the compiler uses
(e.g. how to represent types, traits, and the program itself). It
also contains some amount of the compiler itself, although that is
relatively limited.

Finally, all the crates in the bulge in the middle define the bulk of
the compiler – they all depend on `rustc`, so that they can make use
of the various types defined there, and they export public routines
that `rustc_driver` will invoke as needed (more and more, what these
crates export are "query definitions", but those are covered later
on).

Below `rustc` lie various crates that make up the parser and error
reporting mechanism. For historical reasons, these crates do not have
the `rustc_` prefix, but they are really just as much an internal part
of the compiler and not intended to be stable (though they do wind up
getting used by some crates in the wild; a practice we hope to
gradually phase out).

Each crate has a `README.md` file that describes, at a high-level,
what it contains, and tries to give some kind of explanation (some
better than others).

## The main stages of compilation

The Rust compiler is in a bit of transition right now. It used to be a
purely "pass-based" compiler, where we ran a number of passes over the
entire program, and each did a particular check of transformation. We
are gradually replacing this pass-based code with an alternative setup
based on on-demand **queries**. In the query-model, we work backwards,
executing a *query* that expresses our ultimate goal (e.g. "compile
this crate"). This query in turn may make other queries (e.g. "get me
a list of all modules in the crate"). Those queries make other queries
that ultimately bottom out in the base operations, like parsing the
input, running the type-checker, and so forth. This on-demand model
permits us to do exciting things like only do the minimal amount of
work needed to type-check a single function. It also helps with
incremental compilation. (For details on defining queries, check out
the [query model].)

Regardless of the general setup, the basic operations that the
compiler must perform are the same. The only thing that changes is
whether these operations are invoked front-to-back, or on demand.  In
order to compile a Rust crate, these are the general steps that we
take:

1. **Parsing input**
    - this processes the `.rs` files and produces the AST
      ("abstract syntax tree")
    - the AST is defined in `syntax/ast.rs`. It is intended to match the lexical
      syntax of the Rust language quite closely.
2. **Name resolution, macro expansion, and configuration**
    - once parsing is complete, we process the AST recursively, resolving
      paths and expanding macros. This same process also processes `#[cfg]`
      nodes, and hence may strip things out of the AST as well.
3. **Lowering to HIR**
    - Once name resolution completes, we convert the AST into the HIR,
      or "[high-level intermediate representation]". The HIR is defined in
      `src/librustc/hir/`; that module also includes the [lowering] code.
    - The HIR is a lightly desugared variant of the AST. It is more processed
      than the AST and more suitable for the analyses that follow.
      It is **not** required to match the syntax of the Rust language.
    - As a simple example, in the **AST**, we preserve the parentheses
      that the user wrote, so `((1 + 2) + 3)` and `1 + 2 + 3` parse
      into distinct trees, even though they are equivalent. In the
      HIR, however, parentheses nodes are removed, and those two
      expressions are represented in the same way.
3. **Type-checking and subsequent analyses**
    - An important step in processing the HIR is to perform type
      checking. This process assigns types to every HIR expression,
      for example, and also is responsible for resolving some
      "type-dependent" paths, such as field accesses (`x.f` – we
      can't know what field `f` is being accessed until we know the
      type of `x`) and associated type references (`T::Item` – we
      can't know what type `Item` is until we know what `T` is).
    - Type checking creates "side-tables" (`TypeckTables`) that include
      the types of expressions, the way to resolve methods, and so forth.
    - After type-checking, we can do other analyses, such as privacy checking.
4. **Lowering to MIR and post-processing**
    - Once type-checking is done, we can lower the HIR into MIR ("middle IR"),
      which is a **very** desugared version of Rust, well suited to borrowck
      but also to certain high-level optimizations.
5. **Translation to LLVM and LLVM optimizations**
    - From MIR, we can produce LLVM IR.
    - LLVM then runs its various optimizations, which produces a number of
      `.o` files (one for each "codegen unit").
6. **Linking**
    - Finally, those `.o` files are linked together.


[query model]: query.html
[high-level intermediate representation]: hir.html
[lowering]: lowering.html