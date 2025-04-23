# `*-unknown-hermit`

**Tier: 3**

The [Hermit] unikernel target allows compiling your applications into self-contained, specialized unikernel images that can be run in small virtual machines.

[Hermit]: https://github.com/hermit-os

Target triplets available so far:

- `x86_64-unknown-hermit`
- `aarch64-unknown-hermit`
- `riscv64gc-unknown-hermit`

## Target maintainers

[@stlankes](https://github.com/stlankes)
[@mkroening](https://github.com/mkroening)

## Requirements

These targets only support cross-compilation.
The targets do support std.

When building binaries for this target, the Hermit unikernel is built from scratch.
The application developer themselves specializes the target and sets corresponding expectations.

The Hermit targets follow Linux's `extern "C"` calling convention.

Hermit binaries have the ELF format.

## Building the target

You can build Rust with support for the targets by adding it to the `target` list in `bootstrap.toml`.
To run the Hermit build scripts, you also have to enable your host target.
The build scripts rely on `llvm-tools` and binaries are linked using `rust-lld`, so those have to be enabled as well.

```toml
[build]
build-stage = 1
target = [
    "<HOST_TARGET>",
    "x86_64-unknown-hermit",
    "aarch64-unknown-hermit",
    "riscv64gc-unknown-hermit",
]

[rust]
lld = true
llvm-tools = true
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for these targets.
To compile for these targets, you will either need to build Rust with the targets enabled
(see “Building the targets” above), or build your own copy of `core` by using `build-std` or similar.

As all Hermit programs are unikernels, building a Rust program also requires including the operating system code. A guide for doing so is provided in our starter [hermit-rs-template].

[hermit-rs-template]: https://github.com/hermit-os/hermit-rs-template

## Testing

The targets support running binaries in the form of self-contained unikernel images.
These images can be chainloaded by Hermit's [loader] or hypervisor ([Uhyve]).
QEMU can be used to boot Hermit binaries using the loader on any architecture.
The targets do not support running the Rust test suite.

[loader]: https://github.com/hermit-os/loader
[Uhyve]: https://github.com/hermit-os/uhyve

## Cross-compilation toolchains and C code

The targets do not yet support C code and Rust code at the same time.
