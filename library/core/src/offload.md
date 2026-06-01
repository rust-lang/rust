This module provides support for gpu offloading. For technical details regarding the `offload_kernel`
and how to use it, see their respective documentation.

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

To launch an offloaded kernel, the only current way is to use the `core::intrinsic::offload`
intrinsic (note that intrinsics usage is discouraged outside the standard library). This
allows you to specify grid and block dimensions and pass the required arguments to the device.

```rust,ignore (optional component)
let mut x = [0.0f64; 256];
core::intrinsics::offload::<_, _, ()>(kernel, [256, 1, 1], [1, 1, 1], (&mut x as *mut [f64; 256],));
```

For precise information on the `offload` intrinsic, see its respective documentation.

## Current limitations:

- Usage is restricted to types supported by the current device-mapping implementation.
- Generics and functions accepting dyn Trait are not supported.
- Kernel execution is currently restricted to intrinsics usage, which is discouraged outside of the
standard library.
