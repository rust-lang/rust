# `*-vega-linux-gnu*`

**Tier: 3**

Targets for [Vega OS], Amazon's Linux-based operating system for Fire TV and other devices.

[Vega OS]: https://developer.amazon.com/docs/vega/vega

Target triplets available:

- `aarch64-vega-linux-gnu`
- `armv7-vega-linux-gnueabihf`
- `x86_64-vega-linux-gnu`

## Target maintainers

[@aliiupeng](https://github.com/aliiupeng)

## Requirements

These targets only support cross-compilation. They support `std`.

Vega OS is glibc-based Linux. Binaries built for these targets use the same Linux kernel
interface, glibc ABI, ELF format and `extern "C"` calling convention as the corresponding
`*-unknown-linux-gnu*` targets. They differ in what they expect from the system:

- glibc 2.35 or newer.
- The C/C++ runtime is LLVM's, with no GCC runtime. The system provides LLVM `libunwind`,
  `libc++` and `compiler-rt`, and does not provide `libgcc_s`. `std` for these targets
  therefore links the in-tree LLVM `libunwind` statically. This is the default when the
  targets are built with bootstrap.
- `armv7-vega-linux-gnueabihf` targets Cortex-A7 with NEON and VFPv4, which is what the
  Vega OS userspace itself is built for. `armv7-unknown-linux-gnueabihf` assumes neither.

`target_vendor` is `"vega"`, so code can detect Vega OS with `cfg(target_vendor = "vega")`.
`target_os` is `"linux"` and `target_env` is `"gnu"`, so code written for glibc Linux
works unchanged.

## Building the targets

Add the targets to the `target` list in `bootstrap.toml`. Point each target's C toolchain
at a clang that targets the Vega sysroot, such as the one in the Vega SDK:

```toml
[build]
target = ["aarch64-vega-linux-gnu", "armv7-vega-linux-gnueabihf", "x86_64-vega-linux-gnu"]

[target.aarch64-vega-linux-gnu]
cc = "/path/to/vega-clang-aarch64"
cxx = "/path/to/vega-clang++-aarch64"
ar = "/path/to/llvm-ar"
linker = "/path/to/vega-clang-aarch64"
```

`llvm-libunwind` defaults to `"in-tree"` for these targets.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for these targets. To compile for these
targets, you will either need to build Rust with the targets enabled (see "Building the
targets" above), or build your own copy of `core` by using `build-std` or similar.

The Vega SDK provides a clang toolchain and sysroot for each architecture, and a Cargo
configuration that sets the linker and sysroot. See [Install the Vega SDK].

[Install the Vega SDK]: https://developer.amazon.com/docs/vega/0.21/install-vega-sdk

To link without the SDK's configuration, use clang with `-fuse-ld=lld`,
`-rtlib=compiler-rt`, `-unwindlib=libunwind` and `--sysroot` pointing at a Vega sysroot.

## Testing

`x86_64-vega-linux-gnu` binaries run on the Vega Virtual Device, which is part of the
Vega SDK. `aarch64-vega-linux-gnu` and `armv7-vega-linux-gnueabihf` binaries run on Vega
OS devices, or under `qemu-aarch64` / `qemu-arm` user-mode emulation with `-L` pointing at
a Vega sysroot.

The Rust test suite has not been run against these targets.

## Cross-compilation toolchains and C code

The targets support C and C++ code. Use clang with the Vega sysroot and the flags above.
The Vega SDK's clang and `vega native c-flags <arch>` / `vega native ld-flags <arch>`
provide a ready-made configuration.
