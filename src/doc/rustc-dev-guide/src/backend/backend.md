# The Compiler Backend

The _compiler backend_ refers to the parts of the compiler that turn rustc's
MIR into actual executable code (e.g. an ELF or EXE binary) that can run on a
processor. This is the last stage of compilation, and it has a few important
parts:

0. First, we need to collect the set of things to generate code for. In
   particular, we need to find out which concrete types to substitute for
   generic ones, since we need to generate code for the concrete types.
   Generating code for the concrete types (i.e. emitting a copy of the code for
   each concrete type) is called _monomorphization_, so the process of
   collecting all the concrete types is called _monomorphization collection_.
1. Next, we need to actually lower the MIR (which is generic) to a codegen IR
   (usually LLVM IR; which is not generic) for each concrete type we collected.
2. Finally, we need to invoke LLVM, which runs a bunch of optimization passes,
   generates executable code, and links together an executable binary.
