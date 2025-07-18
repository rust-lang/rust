# std::offload

This module is under active development. Once upstream, it should allow Rust developers to run Rust code on GPUs.
We aim to develop a `rusty` GPU programming interface, which is safe, convenient and sufficiently fast by default.
This includes automatic data movement to and from the GPU, in a efficient way. We will (later)
also offer more advanced, possibly unsafe, interfaces which allow a higher degree of control.

The implementation is based on LLVM's "offload" project, which is already used by OpenMP to run Fortran or C++ code on GPUs.
While the project is under development, users will need to call other compilers like clang to finish the compilation process.
