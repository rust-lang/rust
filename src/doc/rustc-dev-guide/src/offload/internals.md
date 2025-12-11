# std::offload

This module is under active development. Once upstream, it should allow Rust developers to run Rust code on GPUs.
We aim to develop a `rusty` GPU programming interface, which is safe, convenient and sufficiently fast by default.
This includes automatic data movement to and from the GPU, in a efficient way. We will (later)
also offer more advanced, possibly unsafe, interfaces which allow a higher degree of control.

The implementation is based on LLVM's "offload" project, which is already used by OpenMP to run Fortran or C++ code on GPUs.
While the project is under development, users will need to call other compilers like clang to finish the compilation process.

## High-level design:
We use a single-source, two-pass compilation approach. 

First we compile all functions that should be offloaded for the device (e.g nvptx64, amdgcn-amd-amdhsa, intel in the future). Currently we require cumbersome `#cfg(target_os="")` annotations, but we intend to recognize those in the future based on our offload intrinsic. 

We then compile the code for the host (e.g. x86-64), where most of the offloading logic happens. On the host side, we generate calls to the openmp offload runtime, to inform it about the layout of the types (a simplified version of the autodiff TypeTrees). We also use the type system to figure out whether kernel arguments have to be moved only to the device (e.g. `&[f32;1024]`), from the device, or both (e.g. `&mut [f64]`). We then launched the kernel, after which we inform the runtime to end this environment and move data back (as far as needed).
