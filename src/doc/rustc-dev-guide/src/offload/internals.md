# std::offload

This module is under active development.
Once upstream, it should allow Rust developers to run Rust code on GPUs.
We aim to develop a `rusty` GPU programming interface, which is safe, convenient and sufficiently fast by default.
This includes automatic data movement to and from the GPU, in a efficient way.
We will (later) also offer more advanced,
possibly unsafe, interfaces which allow a higher degree of control.

The implementation is based on LLVM's "offload" project,
which is already used by OpenMP to run Fortran or C++ code on GPUs.
While the project is under development,
users will need to call other compilers like clang to finish the compilation process.

## High-level compilation design:

We use a single-source, three-pass compilation approach.

First we compile the host code (e.g. x86-64) to find out which kernel
instantiations the host code requires, including generic ones.
This pass does not perform codegen; instead it writes a manifest that records every
`#[offload_kernel]` instance (with its concrete generic arguments) needed by the host code.

We then compile all functions that should be offloaded for the device
(e.g nvptx64, amdgcn-amd-amdhsa, intel in the future), passing the manifest via
`-Zoffload=Device=<manifest>`.
The recorded kernel instances are added as monomorphization roots, so the required generic
kernels are codegened.
Currently we require cumbersome `#cfg(target_os="")` annotations, but we intend to recognize those in the future based on our offload intrinsic.
This device compilation currently does not leverage rustc's internal Query system, so it will always recompile your kernels at the moment.
This should be easy to fix, but we prioritize features and runtime performance improvements at the moment.
Please reach out if you want to implement it, though!

We then compile the code for the host (e.g. x86-64), where most of the offloading logic happens.
On the host side, we generate calls to the openmp offload runtime,
to inform it about the layout of the types (a simplified version of the autodiff TypeTrees).
We also use the type system to figure out whether kernel arguments have to be moved only to the device (e.g. `&[f32;1024]`),
from the device, or both (e.g. `&mut [f64]`).
We then launch the kernel,
after which we inform the runtime to end this environment and move data back (as far as needed).

The third pass for the host will load the kernel artifacts from the device compilation.
rustc in general may not "guess" or hardcode the build directory layout,
and as such it must be told the paths to the kernel artifacts and the manifest in the respective invocations.
The logic for this could be integrated into cargo,
but it also only requires a trivial cargo wrapper,
which we could trivially provide via crates.io till we see larger adoption.

It might seem tempting to think about a single-source, single pass compilation approach.
However, a lot of the rustc frontend (e.g. AST) will drop any dead code (e.g. code behind an inactive `cfg`).
Getting the frontend to expand and lower code for two targets naively will result in multiple definitions of the same symbol (and other issues).
Trying to teach the whole rustc middle and backend to be aware that any symbol now might contain two implementations is a large undertaking,
and it is questionable why we should make the whole compiler more complex, if the alternative is a ~5 line cargo wrapper.
We still control the full compilation pipeline and have both host and device code available,
therefore there shouldn't be a runtime performance difference between the two approaches.

## Safety and region design

The `Region` type (`core::offload::Region`) is a host-side handle to a contiguous memory region
that should be partitioned across the execution units (threads) of an offloaded kernel. Creating a
`Region` takes a mutable borrow of the underlying slice, so Rust's borrow checker prevents the host
from accessing the memory while the region is alive. `Region` is deliberately neither `Copy` nor
`Clone`, which avoids aliasing the same memory through multiple handles. A region can be reborrowed
with `Region::reborrow` when a shorter lifetime is needed.

The way a region is split across execution units is described by the `PartitioningStrategy` trait.
Given the region pointer and length, the strategy's `get` and `get_mut` methods return a read-only
or mutable view for the current execution unit, or `None` where the current unit has no work to do.
`PartitioningStrategy` is an `unsafe` trait: an implementation must guarantee that all views it
hands out are disjoint, otherwise two units could write to the same location.

A strategy exposes `check_launch(len, grid, block)`, which decides whether a given number of elements
can be launched with a particular grid/block configuration. This check is soundness-critical: it may
only return `Ok` unconditionally if and only if a launch is valid for arbitrary `len`, `grid`, and
`block` values. If any configuration would produce incorrect results or cause undefined behavior, it
must return a `LaunchError`. The `offload!` macro calls `check_launch` for every `Region` argument
before launching, and panics if the check fails.

Only `Region`s passed to a kernel by value can be mapped like slices. A `Region` nested inside another
type is rejected during type checking (see `contains_nested_offload_region` in `rustc_hir_typeck`).
