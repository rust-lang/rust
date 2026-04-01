# aarch64_be-unknown-none-softfloat

**Tier: 3**

Target for freestanding/bare-metal big-endian ARM64 binaries in ELF format:
firmware, kernels, etc.

## Target maintainers

[@Gelbpunkt](https://github.com/Gelbpunkt)

## Requirements

This target is cross-compiled. There is no support for `std`. There is no
default allocator, but it's possible to use `alloc` by supplying an allocator.

The target does not assume existence of a FPU and does not make use of any
non-GPR register. This allows the generated code to run in environments, such
as kernels, which may need to avoid the use of such registers or which may have
special considerations about the use of such registers (e.g. saving and
restoring them to avoid breaking userspace code using the same registers). You
can change code generation to use additional CPU features via the
`-C target-feature=` codegen options to rustc, or via the `#[target_feature]`
mechanism within Rust code.

By default, code generated with the soft-float target should run on any
big-endian ARM64 hardware, enabling additional target features may raise this
baseline.

`extern "C"` uses the [architecture's standard calling convention][aapcs64].

[aapcs64]: https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst

The targets generate binaries in the ELF format. Any alternate formats or
special considerations for binary layout will require linker options or linker
scripts.

## Building the target

You can build Rust with support for the target by adding it to the `target`
list in `bootstrap.toml`:

```toml
[build]
target = ["aarch64_be-unknown-none-softfloat"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will first need to build Rust with the target enabled (see
"Building the target" above).

## Cross-compilation

For cross builds, you will need an appropriate ARM64 C/C++ toolchain for
linking, or if you want to compile C code along with Rust (such as for Rust
crates with C dependencies).

Rust *may* be able to use an `aarch64_be-unknown-linux-{gnu,musl}-` toolchain
with appropriate standalone flags to build for this target (depending on the
assumptions of that toolchain, see below), or you may wish to use a separate
`aarch64_be-unknown-none-softfloat` toolchain.

On some ARM64 hosts that use ELF binaries, you *may* be able to use the host C
toolchain, if it does not introduce assumptions about the host environment that
don't match the expectations of a standalone environment. Otherwise, you may
need a separate toolchain for standalone/freestanding development, just as when
cross-compiling from a non-ARM64 platform.

## Testing

As the target supports a variety of different environments and does not support
`std`, it does not support running the Rust test suite.
