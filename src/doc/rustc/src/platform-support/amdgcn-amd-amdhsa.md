# `amdgcn-amd-amdhsa`

**Tier: 3**

AMD GPU target for compute/HSA (Heterogeneous System Architecture).

## Target maintainers

[@Flakebi](https://github.com/Flakebi)

## Requirements

AMD GPUs can be targeted via cross-compilation.
Supported GPUs depend on the LLVM version that is used by Rust.
In general, most GPUs starting from gfx7 (Sea Islands/CI) are supported as compilation targets, though older GPUs are not supported by the latest host runtime.
Details about supported GPUs can be found in [LLVM’s documentation] and [ROCm documentation].

Binaries can be loaded by [HIP] or by the HSA runtime implemented in [ROCR-Runtime].
The format of binaries is a linked ELF.

Binaries must be built with no-std.
They can use `core` and `alloc` (`alloc` only if an allocator is supplied).
At least one function needs to use the `"gpu-kernel"` calling convention and should be marked with `no_mangle` for simplicity.
Functions using the `"gpu-kernel"` calling convention are kernel entrypoints and can be used from the host runtime.

## Building the target

The target is included in rustc.

## Building Rust programs

The amdgpu target supports many hardware generations, which need different binaries.
The generations are exposed as different target-cpus in the backend.
As there are many, Rust does not ship pre-compiled libraries for this target.
Therefore, you have to build your own copy of `core` by using `cargo -Zbuild-std=core` or similar.

To build a binary, create a no-std library:
```rust,ignore (platform-specific)
// src/lib.rs
#![feature(abi_gpu_kernel)]
#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub extern "gpu-kernel" fn kernel(/* Arguments */) {
    // Code
}
```

Build the library as `cdylib`:
```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[profile.dev]
lto = true # LTO must be explicitly enabled for now
[profile.release]
lto = true
```

The target-cpu must be from the list [supported by LLVM] (or printed with `rustc --target amdgcn-amd-amdhsa --print target-cpus`).
The GPU version on the current system can be found e.g. with [`rocminfo`].

Example `.cargo/config.toml` file to set the target and GPU generation:
```toml
# .cargo/config.toml
[build]
target = "amdgcn-amd-amdhsa"
rustflags = ["-Ctarget-cpu=gfx1100"]

[unstable]
build-std = ["core"] # Optional: "alloc"
```

## Running Rust programs

To run a binary on an AMD GPU, a host runtime is needed.
On Linux and Windows, [HIP] can be used to load and run binaries.
Example code on how to load a compiled binary and run it is available in [ROCm examples].

On Linux, binaries can also run through the HSA runtime as implemented in [ROCR-Runtime].

<!-- Mention an allocator once a suitable one exists for amdgpu -->

<!--
## Testing

Does the target support running binaries, or do binaries have varying
expectations that prevent having a standard way to run them? If users can run
binaries, can they do so in some common emulator, or do they need native
hardware? Does the target support running the Rust testsuite?

-->

## Additional information

More information can be found on the [LLVM page for amdgpu].

[LLVM’s documentation]: https://llvm.org/docs/AMDGPUUsage.html#processors
[ROCm documentation]: https://rocmdocs.amd.com
[HIP]: https://rocm.docs.amd.com/projects/HIP/
[ROCR-Runtime]: https://github.com/ROCm/ROCR-Runtime
[supported by LLVM]: https://llvm.org/docs/AMDGPUUsage.html#processors
[LLVM page for amdgpu]: https://llvm.org/docs/AMDGPUUsage.html
[`rocminfo`]: https://github.com/ROCm/rocminfo
[ROCm examples]: https://github.com/ROCm/rocm-examples/tree/ca8ef5b6f1390176616cd1c18fbc98785cbc73f6/HIP-Basic/module_api
