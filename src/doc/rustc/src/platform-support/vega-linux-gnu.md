# `*-vega-linux-gnu*`

**Tier: 3**

Targets for [Vega OS], Amazon's Linux-based OS for Fire TV and other devices.

[Vega OS]: https://developer.amazon.com/docs/vega/vega

Target triplets available:

- `aarch64-vega-linux-gnu`
- `armv7-vega-linux-gnueabihf`
- `x86_64-vega-linux-gnu`

## Target maintainers

[@aliiupeng](https://github.com/aliiupeng)
[@altrieu-amazon](https://github.com/altrieu-amazon)

## Requirements

These targets only support cross-compilation. They support `std`.

Vega OS is glibc-based Linux. Binaries built for these targets use the same Linux kernel
interface, glibc ABI, ELF format and `extern "C"` calling convention as the corresponding
`*-unknown-linux-gnu*` targets.

- glibc 2.35 or newer.
- There is no shared `libgcc_s`, so `std` for these targets links LLVM `libunwind`
  statically: the in-tree copy when built with bootstrap (the default), or the Vega
  sysroot's `libunwind.a` with `build-std`.
- `armv7-vega-linux-gnueabihf` targets Cortex-A7 with NEON and VFPv4, which is what the
  Vega OS userspace itself is built for. `armv7-unknown-linux-gnueabihf` assumes neither.

The `cfg` values are the same as for the corresponding `*-unknown-linux-gnu*` targets but
with `target_vendor = "vega"`, so code written that only checks os("linux") and env
("gnu") for glibc Linux works as before.

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
targets" above), or build your own copy of `std` by using `build-std` or similar.

The Vega SDK provides a clang toolchain and sysroot for each architecture, and a Cargo
configuration that sets the linker and sysroot. See [Install the Vega SDK].

[Install the Vega SDK]: https://developer.amazon.com/docs/vega/0.24/install-vega-sdk

To link without the SDK's configuration, use clang with these flags: `--target=<llvm-triple>
--sysroot=<sysroot> -fuse-ld=lld -rtlib=compiler-rt -B<gcc-dir> -L<gcc-dir>`

`<llvm-triple>` is the matching upstream triple (`aarch64-unknown-linux-gnu`,
`armv7-unknown-linux-gnueabihf` or `x86_64-unknown-linux-gnu`).
`<gcc-dir>` is the sysroot directory that holds GCC's `crtbeginS.o` and `libgcc.a`, which
the sysroot keeps in an OpenEmbedded-style path,
`<sysroot>/usr/lib/<oe-triple>/<gcc-version>/`.

## Testing

`x86_64-vega-linux-gnu` binaries run on a Linux x86_64 host using the host's glibc.
`aarch64-vega-linux-gnu` and `armv7-vega-linux-gnueabihf` binaries run under
`qemu-aarch64` / `qemu-arm` user-mode emulation with `-L` pointing at a Vega sysroot.

Binaries can also run on the Vega Virtual Device (in the Vega SDK) when the binary's
architecture matches the host's, or on Fire TV devices running Vega OS.

The Rust test suite has not been run against these targets.

## Cross-compilation toolchains and C code

The targets support C and C++ code. Compile it with clang using the same `--target` and
`--sysroot` as above. C++ code uses the sysroot's `libc++`.
When Cargo builds C code through the `cc` crate, set these for each target (below is
an example for `aarch64-vega-linux-gnu`):

```sh
CC_aarch64_vega_linux_gnu=/path/to/clang
AR_aarch64_vega_linux_gnu=/path/to/llvm-ar
CFLAGS_aarch64_vega_linux_gnu="--sysroot=<vega-sysroot-aarch64>"
```

`cc` passes `--target=<Rust target name>` to clang, which accepts the `vega` vendor.
For `armv7-vega-linux-gnueabihf`, also add `-mcpu=cortex-a7 -mfpu=neon-vfpv4` to
`CFLAGS_armv7_vega_linux_gnueabihf`.
