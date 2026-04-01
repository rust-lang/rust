# Code generation

Code generation (or "codegen") is the part of the compiler
that actually generates an executable binary.
Usually, rustc uses LLVM for code generation,
but there is also support for [Cranelift] and [GCC].
The key is that rustc doesn't implement codegen itself.
It's worth noting, though, that in the Rust source code,
many parts of the backend have `codegen` in their names
(there are no hard boundaries).

[Cranelift]: https://github.com/bytecodealliance/wasmtime/tree/main/cranelift
[GCC]: https://github.com/rust-lang/rustc_codegen_gcc

> NOTE: If you are looking for hints on how to debug code generation bugs,
> please see [this section of the debugging chapter][debugging].

[debugging]: ./debugging.md

## What is LLVM?

[LLVM](https://llvm.org) is "a collection of modular and reusable compiler and
toolchain technologies". In particular, the LLVM project contains a pluggable
compiler backend (also called "LLVM"), which is used by many compiler projects,
including the `clang` C compiler and our beloved `rustc`.

LLVM takes input in the form of LLVM IR. It is basically assembly code with
additional low-level types and annotations added. These annotations are helpful
for doing optimizations on the LLVM IR and outputted machine code. The end
result of all this is (at long last) something executable (e.g. an ELF object,
an EXE, or wasm).

There are a few benefits to using LLVM:

- We don't have to write a whole compiler backend. This reduces implementation
  and maintenance burden.
- We benefit from the large suite of advanced optimizations that the LLVM
  project has been collecting.
- We can automatically compile Rust to any of the platforms for which LLVM has
  support. For example, as soon as LLVM added support for wasm, voila! rustc,
  clang, and a bunch of other languages were able to compile to wasm! (Well,
  there was some extra stuff to be done, but we were 90% there anyway).
- We and other compiler projects benefit from each other. For example, when the
  [Spectre and Meltdown security vulnerabilities][spectre] were discovered,
  only LLVM needed to be patched.

[spectre]: https://meltdownattack.com/

## Running LLVM, linking, and metadata generation

Once LLVM IR for all of the functions and statics, etc is built, it is time to
start running LLVM and its optimization passes. LLVM IR is grouped into
"modules". Multiple "modules" can be codegened at the same time to aid in
multi-core utilization. These "modules" are what we refer to as _codegen
units_. These units were established way back during monomorphization
collection phase.

Once LLVM produces objects from these modules, these objects are passed to the
linker along with, optionally, the metadata object and an archive or an
executable is produced.

It is not necessarily the codegen phase described above that runs the
optimizations. With certain kinds of LTO, the optimization might happen at the
linking time instead. It is also possible for some optimizations to happen
before objects are passed on to the linker and some to happen during the
linking.

This all happens towards the very end of compilation. The code for this can be
found in [`rustc_codegen_ssa::back`][ssaback] and
[`rustc_codegen_llvm::back`][llvmback]. Sadly, this piece of code is not
really well-separated into LLVM-dependent code; the [`rustc_codegen_ssa`][ssa]
contains a fair amount of code specific to the LLVM backend.

Once these components are done with their work you end up with a number of
files in your filesystem corresponding to the outputs you have requested.

[ssa]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/index.html
[ssaback]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/back/index.html
[llvmback]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_llvm/back/index.html
