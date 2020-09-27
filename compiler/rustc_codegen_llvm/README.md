The `codegen` crate contains the code to convert from MIR into LLVM IR,
and then from LLVM IR into machine code. In general it contains code
that runs towards the end of the compilation process.

For more information about how codegen works, see the [rustc dev guide].

[rustc dev guide]: https://rustc-dev-guide.rust-lang.org/backend/codegen.html
