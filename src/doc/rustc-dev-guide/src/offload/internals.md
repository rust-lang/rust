# std::offload

This module is under active development. Once upstream, it should allow Rust developers to run Rust code on GPUs.
We aim to develop a `rusty` GPU programming interface, which is safe, convenient and sufficiently fast by default.
This includes automatic data movement to and from the GPU, in a efficient way. We will (later)
also offer more advanced, possibly unsafe, interfaces which allow a higher degree of control.

The implementation is based on LLVM's "offload" project, which is already used by OpenMP to run Fortran or C++ code on GPUs.
While the project is under development, users will need to call other compilers like clang to finish the compilation process.

## High-level compilation design:
We use a single-source, two-pass compilation approach. 

First we compile all functions that should be offloaded for the device (e.g nvptx64, amdgcn-amd-amdhsa, intel in the future). Currently we require cumbersome `#cfg(target_os="")` annotations, but we intend to recognize those in the future based on our offload intrinsic. 
This first compilation currently does not leverage rustc's internal Query system, so it will always recompile your kernels at the moment. This should be easy to fix, but we prioritize features and runtime performance improvements at the moment. Please reach out if you want to implement it, though!

We then compile the code for the host (e.g. x86-64), where most of the offloading logic happens. On the host side, we generate calls to the openmp offload runtime, to inform it about the layout of the types (a simplified version of the autodiff TypeTrees). We also use the type system to figure out whether kernel arguments have to be moved only to the device (e.g. `&[f32;1024]`), from the device, or both (e.g. `&mut [f64]`). We then launch the kernel, after which we inform the runtime to end this environment and move data back (as far as needed).

The second pass for the host will load the kernel artifacts from the previous compilation. rustc in general may not "guess" or hardcode the build directory layout, and as such it must be told the path to the kernel artifacts in the second invocation. The logic for this could be integrated into cargo, but it also only requires a trivial cargo wrapper, which we could trivially provide via crates.io till we see larger adoption.

It might seem tempting to think about a single-source, single pass compilation approach. However, a lot of the rustc frontend (e.g. AST) will drop any dead code (e.g. code behind an inactive `cfg`). Getting the frontend to expand and lower code for two targets naively will result in multiple definitions of the same symbol (and other issues). Trying to teach the whole rustc middle and backend to be aware that any symbol now might contain two implementations is a large undertaking, and it is questionable why we should make the whole compiler more complex, if the alternative is a ~5 line cargo wrapper. We still control the full compilation pipeline and have both host and device code available, therefore there shouldn't be a runtime performance difference between the two approaches. 

