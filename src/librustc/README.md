An informal guide to reading and working on the rustc compiler.
==================================================================

If you wish to expand on this document, or have a more experienced
Rust contributor add anything else to it, please get in touch:

* https://internals.rust-lang.org/
* https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust

or file a bug:

https://github.com/rust-lang/rust/issues

Your concerns are probably the same as someone else's.

You may also be interested in the
[Rust Forge](https://forge.rust-lang.org/), which includes a number of
interesting bits of information.

Finally, at the end of this file is a GLOSSARY defining a number of
common (and not necessarily obvious!) names that are used in the Rust
compiler code. If you see some funky name and you'd like to know what
it stands for, check there!

The crates of rustc
===================

Rustc consists of a number of crates, including `syntax`,
`rustc`, `rustc_back`, `rustc_trans`, `rustc_driver`, and
many more. The source for each crate can be found in a directory
like `src/libXXX`, where `XXX` is the crate name.

(NB. The names and divisions of these crates are not set in
stone and may change over time -- for the time being, we tend towards
a finer-grained division to help with compilation time, though as
incremental improves that may change.)

The dependency structure of these crates is roughly a diamond:

````
                  rustc_driver
                /      |       \
              /        |         \
            /          |           \
          /            v             \
rustc_trans    rustc_borrowck   ...  rustc_metadata
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


The idea is that `rustc_driver`, at the top of this lattice, basically
defines the overall control-flow of the compiler. It doesn't have much
"real code", but instead ties together all of the code defined in the
other crates and defines the overall flow of execution.

At the other extreme, the `rustc` crate defines the common and
pervasive data structures that all the rest of the compiler uses
(e.g., how to represent types, traits, and the program itself). It
also contains some amount of the compiler itself, although that is
relatively limited.

Finally, all the crates in the bulge in the middle define the bulk of
the compiler -- they all depend on `rustc`, so that they can make use
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

The compiler process
====================

The Rust compiler is comprised of six main compilation phases.

1. Parsing input
2. Configuration & expanding (cfg rules & syntax extension expansion)
3. Running analysis passes
4. Translation to LLVM
5. LLVM passes
6. Linking

Phase one is responsible for parsing & lexing the input to the compiler. The
output of this phase is an abstract syntax tree (AST). The AST at this point
includes all macro uses & attributes. This means code which will be later
expanded and/or removed due to `cfg` attributes is still present in this
version of the AST. Parsing abstracts away details about individual files which
have been read into the AST.

Phase two handles configuration and macro expansion. You can think of this
phase as a function acting on the AST from the previous phase. The input for
this phase is the unexpanded AST from phase one, and the output is an expanded
version of the same AST. This phase will expand all macros & syntax
extensions and will evaluate all `cfg` attributes, potentially removing some
code. The resulting AST will not contain any macros or `macro_use` statements.

The code for these first two phases is in [`libsyntax`][libsyntax].

After this phase, the compiler allocates IDs to each node in the AST
(technically not every node, but most of them). If we are writing out
dependencies, that happens now.

The third phase is analysis. This is the most complex phase in the compiler,
and makes up much of the code. This phase included name resolution, type
checking, borrow checking, type & lifetime inference, trait selection, method
selection, linting and so on. Most of the error detection in the compiler comes
from this phase (with the exception of parse errors which arise during
parsing). The "output" of this phase is a set of side tables containing
semantic information about the source program. The analysis code is in
[`librustc`][rustc] and some other crates with the `librustc_` prefix.

The fourth phase is translation. This phase translates the AST (and the side
tables from the previous phase) into LLVM IR (intermediate representation).
This is achieved by calling into the LLVM libraries. The code for this is in
[`librustc_trans`][trans].

Phase five runs the LLVM backend. This runs LLVM's optimization passes on the
generated IR and generates machine code resulting in object files. This phase
is not really part of the Rust compiler, as LLVM carries out all the work.
The interface between LLVM and Rust is in [`librustc_llvm`][llvm].

The final phase, phase six, links the object files into an executable. This is
again outsourced to other tools and not performed by the Rust compiler
directly. The interface is in [`librustc_back`][back] (which also contains some
things used primarily during translation).

A module called the driver coordinates all these phases. It handles all the
highest level coordination of compilation from parsing command line arguments
all the way to invoking the linker to produce an executable.

Modules in the librustc crate
=============================

The librustc crate itself consists of the following submodules
(mostly, but not entirely, in their own directories):

- session: options and data that pertain to the compilation session as
  a whole
- middle: middle-end: name resolution, typechecking, LLVM code
  generation
- metadata: encoder and decoder for data required by separate
  compilation
- plugin: infrastructure for compiler plugins
- lint: infrastructure for compiler warnings
- util: ubiquitous types and helper functions
- lib: bindings to LLVM

The entry-point for the compiler is main() in the [`librustc_driver`][driver]
crate.

The 3 central data structures:
------------------------------

1. `./../libsyntax/ast.rs` defines the AST. The AST is treated as
   immutable after parsing, but it depends on mutable context data
   structures (mainly hash maps) to give it meaning.

   - Many – though not all – nodes within this data structure are
     wrapped in the type `spanned<T>`, meaning that the front-end has
     marked the input coordinates of that node. The member `node` is
     the data itself, the member `span` is the input location (file,
     line, column; both low and high).

   - Many other nodes within this data structure carry a
     `def_id`. These nodes represent the 'target' of some name
     reference elsewhere in the tree. When the AST is resolved, by
     `middle/resolve.rs`, all names wind up acquiring a def that they
     point to. So anything that can be pointed-to by a name winds
     up with a `def_id`.

2. `middle/ty.rs` defines the datatype `sty`. This is the type that
   represents types after they have been resolved and normalized by
   the middle-end. The typeck phase converts every ast type to a
   `ty::sty`, and the latter is used to drive later phases of
   compilation. Most variants in the `ast::ty` tag have a
   corresponding variant in the `ty::sty` tag.

3. `./../librustc_llvm/lib.rs` defines the exported types
   `ValueRef`, `TypeRef`, `BasicBlockRef`, and several others.
   Each of these is an opaque pointer to an LLVM type,
   manipulated through the `lib::llvm` interface.

[libsyntax]: https://github.com/rust-lang/rust/tree/master/src/libsyntax/
[trans]: https://github.com/rust-lang/rust/tree/master/src/librustc_trans/
[llvm]: https://github.com/rust-lang/rust/tree/master/src/librustc_llvm/
[back]: https://github.com/rust-lang/rust/tree/master/src/librustc_back/
[rustc]: https://github.com/rust-lang/rust/tree/master/src/librustc/
[driver]: https://github.com/rust-lang/rust/tree/master/src/librustc_driver

Glossary
========

The compiler uses a number of...idiosyncratic abbreviations and
things. This glossary attempts to list them and give you a few
pointers for understanding them better.

- AST -- the **abstract syntax tree** produced the `syntax` crate; reflects user syntax
  very closely.
- cx -- we tend to use "cx" as an abbrevation for context. See also tcx, infcx, etc.
- HIR -- the **High-level IR**, created by lowering and desugaring the AST. See `librustc/hir`.
- `'gcx` -- the lifetime of the global arena (see `librustc/ty`).
- generics -- the set of generic type parameters defined on a type or item
- infcx -- the inference context (see `librustc/infer`)
- MIR -- the **Mid-level IR** that is created after type-checking for use by borrowck and trans.
  Defined in the `src/librustc/mir/` module, but much of the code that manipulates it is
  found in `src/librustc_mir`.
- obligation -- something that must be proven by the trait system.
- sess -- the **compiler session**, which stores global data used throughout compilation
- substs -- the **substitutions** for a given generic type or item
  (e.g., the `i32, u32` in `HashMap<i32, u32>`)
- tcx -- the "typing context", main data structure of the compiler (see `librustc/ty`).
- trans -- the code to **translate** MIR into LLVM IR.
- trait reference -- a trait and values for its type parameters (see `librustc/ty`).
- ty -- the internal representation of a **type** (see `librustc/ty`).
