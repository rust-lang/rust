# Code generation

Code generation or "codegen" is the part of the compiler that actually
generates an executable binary. rustc uses LLVM for code generation.

> NOTE: If you are looking for hints on how to debug code generation bugs,
> please see [this section of the debugging chapter][debugging].

[debugging]: codegen/debugging.html

## What is LLVM?

All of the preceding chapters of this guide have one thing in common: we never
generated any executable machine code at all! With this chapter, all of that
changes.

Like most compilers, rustc is composed of a "frontend" and a "backend". The
"frontend" is responsible for taking raw source code, checking it for
correctness, and getting it into a format `X` from which we can generate
executable machine code. The "backend" then takes that format `X` and produces
(possibly optimized) executable machine code for some platform. All of the
previous chapters deal with rustc's frontend.

rustc's backend is [LLVM](https://llvm.org), "a collection of modular and
reusable compiler and toolchain technologies". In particular, the LLVM project
contains a pluggable compiler backend (also called "LLVM"), which is used by
many compiler projects, including the `clang` C compiler and our beloved
`rustc`.

LLVM's "format `X`" is called LLVM IR. It is basically assembly code with
additional low-level types and annotations added. These annotations are helpful
for doing optimizations on the LLVM IR and outputted machine code. The end
result of all this is (at long last) something executable (e.g. an ELF object
or wasm).

There are a few benefits to using LLVM:

- We don't have to write a whole compiler backend. This reduces implementation
  and maintenance burden.
- We benefit from the large suite of advanced optimizations that the LLVM
  project has been collecting.
- We automatically can compile Rust to any of the platforms for which LLVM has
  support. For example, as soon as LLVM added support for wasm, voila! rustc,
  clang, and a bunch of other languages were able to compile to wasm! (Well,
  there was some extra stuff to be done, but we were 90% there anyway).
- We and other compiler projects benefit from each other. For example, when the
  [Spectre and Meltdown security vulnerabilities][spectre] were discovered,
  only LLVM needed to be patched.

[spectre]: https://meltdownattack.com/

## Generating LLVM IR

TODO
