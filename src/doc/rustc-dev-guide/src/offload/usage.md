# Usage

This feature is work-in-progress, and not ready for usage.
The instructions here are for contributors, or people interested in following the latest progress.
We currently work on launching the following Rust kernel on the GPU.
To follow along, copy it to a `src/lib.rs` file.

```rust
#![allow(internal_features)]
#![feature(gpu_offload)]
#![cfg_attr(target_os = "linux", feature(core_intrinsics))]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu, abi_gpu_kernel))]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx, abi_gpu_kernel))]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

use core::offload::offload_kernel;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{workgroup_id_x as block_idx_x, workitem_id_x as thread_idx_x};
#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};

// Kernels can be generic, like any other Rust function.
// The concrete instantiations required by the host code are collected in a
// manifest, which the device compilation then reads (see below).
#[offload_kernel]
fn kernel<T: Copy>(x: *mut [T; 256], value: T) {
    unsafe {
        let n = (*x).len();
        let i = (thread_idx_x() + block_idx_x() * block_dim_x()) as usize;
        if i < n {
            (*x)[i] = value;
        }
    }
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
fn main() {
    let mut x = [0.0f64; 256];
    core::offload::offload! {
        kernel = kernel,
        workgroup_dim = [256, 1, 1],
        args = (&mut x as *mut [f64; 256], 2.5),
    }
    for i in 0..x.len() {
        assert_eq!(x[i], 2.5);
    }
    unsafe { libc::printf(c"all checks passed".as_ptr()); }
}
```

## Compile instructions
It is important to use a clang compiler build on the same LLVM as rustc.
Just calling clang without the full path will likely use your system clang, which probably will be incompatible.
So either substitute clang/lld invocations below with absolute path, or set your `PATH` accordingly.

The compilation runs three passes:
1. `HostMetadata`: compile the host code, writing a manifest that lists the kernel
   instantiations (including generic ones) required by the host code.
2. `Device`: compile the kernels for the GPU, reading the manifest so the recorded generic
   instantiations are codegened.
3. `Host`: generate the final host code, embedding the device artifact.

<div class="warning">

Replace the `target-cpu` (gfx90a) with the right code for your GPU.
These are often referred to as "LLVM target names"[^list].

</div>

First we generate the manifest from the host code:
```
RUSTFLAGS="--emit=llvm-bc,llvm-ir -Csave-temps -Zoffload=HostMetadata=/absolute/path/to/offload.manifest -Zunstable-options" cargo +offload build -r
```
This pass only writes the manifest.

Now we generate the device (GPU) code, passing the manifest:
```
RUSTFLAGS="-Ctarget-cpu=gfx90a --emit=llvm-bc,llvm-ir -Zoffload=Device=/absolute/path/to/offload.manifest -Csave-temps -Zunstable-options" cargo +offload build -Zunstable-options -r -v --target amdgcn-amd-amdhsa -Zbuild-std=core
```

Next we generate the host (CPU) code.
```
RUSTFLAGS="--emit=llvm-bc,llvm-ir -Csave-temps -Zoffload=Host=$PWD/target/amdgcn-amd-amdhsa/release/deps/device.bin -Zunstable-options" cargo +offload build -r
```

In the final step, you can now run your binary

```
LD_LIBRARY_PATH=$(rustc +nightly --print sysroot)/lib  ./target/x86_64-unknown-linux-gnu/release/binary-name
all checks passed!
```

These three steps will soon be wrapped into a single command, once we had more time to test all steps.

To receive more information about the memory transfer, you can enable info printing by adding `LIBOMPTARGET_INFO=-1` ahead of your binary call.

[^list]: https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html or https://developer.nvidia.com/cuda/gpus. Alternatively, check `rustc --print target-cpus`.
