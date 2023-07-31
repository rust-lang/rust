# `no-parallel-llvm`

---------------------

This flag disables parallelization of codegen and linking, while otherwise preserving
behavior with regard to codegen units and LTO.

This flag is not useful for regular users, but it can be useful for debugging the backend. Codegen issues commonly only manifest under specific circumstances, e.g. if multiple codegen units are used and ThinLTO is enabled. Serialization of these threaded configurations makes the use of LLVM debugging facilities easier, by avoiding the interleaving of output.
