An informal guide to reading and working on the rustc compiler.
==================================================================

If you wish to expand on this document, or have a more experienced
Rust contributor add anything else to it, please get in touch:

* https://internals.rust-lang.org/
* https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust

or file a bug:

https://github.com/rust-lang/rust/issues

Your concerns are probably the same as someone else's.

The crates of rustc
===================

Rustc consists of a number of crates, including `libsyntax`,
`librustc`, `librustc_back`, `librustc_trans`, and `librustc_driver`
(the names and divisions are not set in stone and may change;
in general, a finer-grained division of crates is preferable):

- [`libsyntax`][libsyntax] contains those things concerned purely with syntax –
  that is, the AST, parser, pretty-printer, lexer, macro expander, and
  utilities for traversing ASTs – are in a separate crate called
  "syntax", whose files are in `./../libsyntax`, where `.` is the
  current directory (that is, the parent directory of front/, middle/,
  back/, and so on).

- `librustc` (the current directory) contains the high-level analysis
  passes, such as the type checker, borrow checker, and so forth.
  It is the heart of the compiler.

- [`librustc_back`][back] contains some very low-level details that are
  specific to different LLVM targets and so forth.

- [`librustc_trans`][trans] contains the code to convert from Rust IR into LLVM
  IR, and then from LLVM IR into machine code, as well as the main
  driver that orchestrates all the other passes and various other bits
  of miscellany. In general it contains code that runs towards the
  end of the compilation process.

- [`librustc_driver`][driver] invokes the compiler from
  [`libsyntax`][libsyntax], then the analysis phases from `librustc`, and
  finally the lowering and codegen passes from [`librustc_trans`][trans].

Roughly speaking the "order" of the three crates is as follows:

              librustc_driver
                      |
    +-----------------+-------------------+
    |                                     |
    libsyntax -> librustc -> librustc_trans


The compiler process:
=====================

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
