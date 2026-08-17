This module provides support for gpu offloading. For technical details regarding the `offload_kernel`
and `offload!` macros, see their respective documentation.

## General usage
The `offload_kernel` macro can be applied to a function to generate the necessary code to launch a
kernel on the target device.

```rust,ignore (optional component)
#[offload_kernel]
fn kernel(x: *mut [f64; 256]) {
    // SAFETY:
    // calling our `arch` functions and dereferencing a raw pointer is unsafe
    unsafe {
        let n = (*x).len();
        let i = (thread_idx_x() + block_idx_x() * block_dim_x()) as usize;
        if i < n {
            (*x)[i] = i as f64;
        }
    }
}
```

To launch an offloaded kernel, use the `offload!` macro. It lets you specify the kernel, the
workgroup and thread dimensions, and the arguments to forward to the device.

```rust,ignore (optional component)
let mut x = [0.0f64; 256];
core::offload::offload! {
    kernel = kernel,
    workgroup_dim = [256, 1, 1],
    args = (&mut x as *mut [f64; 256],),
}
```

For precise information on the underlying `offload` intrinsic, see its respective documentation.

## Current limitations:

- Usage is restricted to types supported by the current device-mapping implementation.
- Functions accepting dyn Trait are not supported.
