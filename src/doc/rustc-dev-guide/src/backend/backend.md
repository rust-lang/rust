# The Compiler Backend

All of the preceding chapters of this guide have one thing in common: we never
generated any executable machine code at all! With this chapter, all of that
changes.

It's often useful to think of compilers as being composed of a _frontend_ and a
_backend_  (though in rustc, there's not a sharp line between frontend and
backend). The _frontend_ is responsible for taking raw source code, checking it
for correctness, and getting it into a format usable by the backend. For rustc,
this format is the MIR.  The _backend_ refers to the parts of the compiler that
turn rustc's MIR into actual executable code (e.g. an ELF or EXE binary) that
can run on a processor.  All of the previous chapters deal with rustc's
frontend.

rustc's backend does the following:

0. First, we need to collect the set of things to generate code for. In
   particular, we need to find out which concrete types to substitute for
   generic ones, since we need to generate code for the concrete types.
   Generating code for the concrete types (i.e. emitting a copy of the code for
   each concrete type) is called _monomorphization_, so the process of
   collecting all the concrete types is called _monomorphization collection_.
1. Next, we need to actually lower the MIR to a codegen IR
   (usually LLVM IR) for each concrete type we collected.
2. Finally, we need to invoke the codegen backend (e.g. LLVM or Cranelift),
   which runs a bunch of optimization passes, generates executable code, and
   links together an executable binary.

[codegen1]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/base/fn.codegen_crate.html

The code for codegen is actually a bit complex due to a few factors:

- Support for multiple backends (LLVM and Cranelift). We try to share as much
  backend code between them as possible, so a lot of it is generic over the
  codegen implementation. This means that there are often a lot of layers of
  abstraction.
- Codegen happens asynchronously in another thread for performance.
- The actual codegen is done by a third-party library (either LLVM or Cranelift).

Generally, the [`rustc_codegen_ssa`][ssa] crate contains backend-agnostic code
(i.e. independent of LLVM or Cranelift), while the [`rustc_codegen_llvm`][llvm]
crate contains code specific to LLVM codegen.

[ssa]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/index.html
[llvm]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_llvm/index.html

At a very high level, the entry point is
[`rustc_codegen_ssa::base::codegen_crate`][codegen1]. This function starts the
process discussed in the rest of this chapter.
