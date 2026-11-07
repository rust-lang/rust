# From MIR to binaries

All of the preceding chapters of this guide have one thing in common:
we never generated any executable machine code at all!
With this chapter, all of that changes.

So far,
we've shown how the compiler can take raw source code in text format
and transform it into [MIR].
We have also shown how the compiler does various
analyses on the code to detect things like type or lifetime errors.
Now, we will finally take the MIR and produce some executable machine code.

[MIR]: ./mir/index.md

> NOTE: This part of a compiler is often called the _backend_.
> The term is a bit overloaded because in the compiler source,
> it usually refers to the "codegen backend" (i.e. LLVM, Cranelift, or GCC).
> Usually, when you see the word "backend"  in this part,
> we are referring to the "codegen backend".

So what do we need to do?

1. First, we need to collect the set of things to generate code for.
   In particular,
   we need to find out which concrete types to substitute for generic ones,
   since we need to generate code for the concrete types.
   Generating code for the concrete types
   (i.e. emitting a copy of the code for each concrete type) is called _monomorphization_,
   so the process of collecting all the concrete types is called _monomorphization collection_.
2. Next, we need to actually lower the MIR to a codegen IR
   (usually LLVM IR) for each concrete type we collected.
3. Finally, we need to invoke the codegen backend,
   which runs a bunch of optimization passes,
   generates executable code,
   and links together an executable binary.

[codegen1]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/base/fn.codegen_crate.html

The code for codegen is actually a bit complex due to a few factors:

- Support for multiple codegen backends (LLVM, Cranelift, and GCC).
  We try to share as much backend code between them as possible,
  so a lot of it is generic over the codegen implementation.
  This means that there are often a lot of layers of abstraction.
- Codegen happens asynchronously in another thread for performance.
- The actual codegen is done by a third-party library (either of the 3 backends).

Generally, the [`rustc_codegen_ssa`][ssa] crate contains backend-agnostic code,
while the [`rustc_codegen_llvm`][llvm] crate contains code specific to LLVM codegen.

[ssa]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/index.html
[llvm]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_llvm/index.html

At a very high level, the entry point is
[`rustc_codegen_ssa::base::codegen_crate`][codegen1].
This function starts the process discussed in the rest of this chapter.
