NB: This crate is part of the Rust compiler. For an overview of the
compiler as a whole, see
[the README.md file found in `librustc`](../librustc/README.md).

The `trans` crate contains the code to convert from MIR into LLVM IR,
and then from LLVM IR into machine code. In general it contains code
that runs towards the end of the compilation process.
