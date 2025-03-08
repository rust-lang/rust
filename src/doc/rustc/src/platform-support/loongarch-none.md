# `loongarch*-unknown-none*`

**Tier: 2**

Freestanding/bare-metal LoongArch64 binaries in ELF format: firmware, kernels, etc.

| Target | Description |
|--------|-------------|
| `loongarch64-unknown-none` | LoongArch 64-bit, LP64D ABI (freestanding, hard-float) |
| `loongarch64-unknown-none-softfloat` | LoongArch 64-bit, LP64S ABI (freestanding, soft-float) |

## Target maintainers

- [WANG Rui](https://github.com/heiher) `wangrui@loongson.cn`
- [WANG Xuerui](https://github.com/xen0n) `git@xen0n.name`

## Requirements

This target is cross-compiled. There is no support for `std`. There is no
default allocator, but it's possible to use `alloc` by supplying an allocator.

The `*-softfloat` target does not assume existence of FPU or any other LoongArch
ISA extension, and does not make use of any non-GPR register.
This allows the generated code to run in environments, such as kernels, which
may need to avoid the use of such registers or which may have special considerations
about the use of such registers (e.g. saving and restoring them to avoid breaking
userspace code using the same registers). You can change code generation to use
additional CPU features via the `-C target-feature=` codegen options to rustc, or
via the `#[target_feature]` mechanism within Rust code.

By default, code generated with the soft-float target should run on any
LoongArch64 hardware, with the hard-float target additionally requiring an FPU;
enabling additional target features may raise this baseline.

Code generated with the targets will use the `medium` code model by default.
You can change this using the `-C code-model=` option to rustc.

On `loongarch64-unknown-none*`, `extern "C"` uses the [architecture's standard calling convention][lapcs].

[lapcs]: https://github.com/loongson/la-abi-specs/blob/release/lapcs.adoc

The targets generate binaries in the ELF format. Any alternate formats or
special considerations for binary layout will require linker options or linker
scripts.

## Building the target

You can build Rust with support for the targets by adding them to the `target`
list in `bootstrap.toml`:

```toml
[build]
build-stage = 1
target = [
  "loongarch64-unknown-none",
  "loongarch64-unknown-none-softfloat",
]
```

## Testing

As the targets support a variety of different environments and do not support
`std`, they do not support running the Rust test suite.

## Building Rust programs

Starting with Rust 1.74, precompiled artifacts are provided via `rustup`:

```sh
# install cross-compile toolchain
rustup target add loongarch64-unknown-none
# target flag may be used with any cargo or rustc command
cargo build --target loongarch64-unknown-none
```

## Cross-compilation toolchains and C code

For cross builds, you will need an appropriate LoongArch C/C++ toolchain for
linking, or if you want to compile C code along with Rust (such as for Rust
crates with C dependencies).

Rust *may* be able to use an `loongarch64-unknown-linux-gnu-` toolchain with
appropriate standalone flags to build for this toolchain (depending on the assumptions
of that toolchain, see below), or you may wish to use a separate
`loongarch64-unknown-none` toolchain.

On some LoongArch hosts that use ELF binaries, you *may* be able to use the host
C toolchain, if it does not introduce assumptions about the host environment
that don't match the expectations of a standalone environment. Otherwise, you
may need a separate toolchain for standalone/freestanding development, just as
when cross-compiling from a non-LoongArch platform.
