% Overview

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
version of the AST. Parsing abstracts away details about individual fies which
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
This is achieved by calling into the LLVM libraries rather than writing IR
directly to a file. The code for this is in [`librustc_trans`][trans].

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

The next section of the guide covers the driver & individual phases in more
detail.

[libsyntax]: https://github.com/rust-lang/rust/tree/1.6.0/src/libsyntax/
[trans]: https://github.com/rust-lang/rust/tree/1.6.0/src/librustc_trans/
[llvm]: https://github.com/rust-lang/rust/tree/1.6.0/src/librustc_llvm/
[back]: https://github.com/rust-lang/rust/tree/1.6.0/src/librustc_back/
[rustc]: https://github.com/rust-lang/rust/tree/1.6.0/src/librustc/
