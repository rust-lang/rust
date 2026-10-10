This module provides support for gpu offloading. For technical details regarding the `offload_kernel`
and `offload!` macros, see their respective documentation.

## General usage
The `offload_kernel` macro can be applied to a function to generate the necessary code to launch a
kernel on the target device.

The two targets name the indices differently, and only `nvptx` has them as `unsafe`, hence the
`cfg`.

```rust,ignore (optional component)
#![feature(gpu_offload)]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
use core::offload::offload_kernel;

#[cfg(target_arch = "amdgpu")]
fn global_idx_x() -> u32 {
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    // This has to match the first value of `thread_dim`, because on AMD there is no
    // workgroup-size query. If they disagree the indices break silently: collisions if
    // it is smaller, unwritten gaps if it is larger.
    const THREADS_PER_GROUP: u32 = 64;
    workitem_id_x() + workgroup_id_x() * THREADS_PER_GROUP
}

#[cfg(target_arch = "nvptx64")]
fn global_idx_x() -> u32 {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
    // SAFETY:
    // the `cfg` guarantees we are on nvptx64, where these intrinsics always exist, and
    // they are special-register reads that do not touch memory
    unsafe { _thread_idx_x() + _block_idx_x() * _block_dim_x() }
}

#[offload_kernel]
fn kernel(x: *mut [f64; 256]) {
    // SAFETY:
    // calling our `arch` functions and dereferencing a raw pointer is unsafe
    unsafe {
        let n = (*x).len();
        let i = global_idx_x() as usize;
        if i < n {
            (*x)[i] = i as f64;
        }
    }
}
```

To launch an offloaded kernel, use the `offload!` macro. It lets you specify the kernel, the
workgroup and thread dimensions, the device to offload to, and the arguments to forward to the
device.

```rust,ignore (optional component)
let mut x = [0.0f64; 256];
core::offload::offload! {
    kernel = kernel,
    workgroup_dim = [4, 1, 1],
    thread_dim = [64, 1, 1],
    args = (&mut x as *mut [f64; 256],),
}
```

For precise information on the underlying `offload` intrinsic, see its respective documentation.

## Current limitations:

- Usage is restricted to types supported by the current device-mapping implementation.
- Functions accepting dyn Trait are not supported.
- Thread indices are not portable between targets, and `amdgpu` does not know its own
  workgroup size, so a kernel for both GPUs needs the `cfg` shim and has to repeat
  `thread_dim` as a constant.
