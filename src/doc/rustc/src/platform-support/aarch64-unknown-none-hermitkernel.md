# `aarch64-unknown-none-hermitkernel`

**Tier: 3**

Required to build the kernel for [HermitCore](https://github.com/hermitcore/hermit-playground)
or [RustyHermit](https://github.com/hermitcore/rusty-hermit).
The result is a bare-metal aarch64 binary in ELF format.

## Target maintainers

- Stefan Lankes, https://github.com/stlankes

## Requirements

This target is cross-compiled. There is no support for `std`, but the
library operating system provides a simple allocator to use `alloc`.

By default, Rust code generated for this target does not use any vector or
floating-point registers. This allows the generated code to build the library
operaring system, which may need to avoid the use of such
registers or which may have special considerations about the use of such
registers (e.g. saving and restoring them to avoid breaking userspace code
using the same registers). In contrast to `aarch64-unknown-none-softfloat`,
the target is completly relocatable, which is a required feature of RustyHermit.

By default, code generated with this target should run on any `aarch64`
hardware; enabling additional target features may raise this baseline.
On `aarch64-unknown-none-hermitkernel`, `extern "C"` uses the [standard System V calling
convention](https://github.com/ARM-software/abi-aa/releases/download/2021Q3/sysvabi64.pdf),
without red zones.

This target generated binaries in the ELF format.

## Building the target

Typical you should not use the target directly. The target `aarch64-unknown-hermit`
builds the _user space_ of RustyHermit and supports red zones and floating-point
operations.
To build and link the kernel to the application, the crate
[hermit-sys](https://github.com/hermitcore/rusty-hermit/tree/master/hermit-sys)
should be used by adding the following lines to the `Cargo.toml` file of
your application.

```toml
[target.'cfg(target_os = "hermit")'.dependencies]
hermit-sys = "0.1.*"
```

The crate `hermit-sys` uses the target `aarch64-unknown-none-hermitkernel`
to build the kernel.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you need to build the crate `hermit-sys` (see
"Building the target" above).

## Testing

As `aarch64-unknown-none-hermitkernel` does not support `std`
and does not support running any Rust testsuite.

## Cross-compilation toolchains and C code

If you want to compile C code along with Rust you will need an
appropriate `aarch64` toolchain.

Rust *may* be able to use an `aarch64-linux-gnu-` toolchain with appropriate
standalone flags to build for this toolchain (depending on the assumptions of
that toolchain, see below), or you may wish to use a separate
`aarch64-unknown-none` (or `aarch64-elf-`) toolchain.

On some `aarch64` hosts that use ELF binaries, you *may* be able to use the host
C toolchain, if it does not introduce assumptions about the host environment
that don't match the expectations of a standalone environment. Otherwise, you
may need a separate toolchain for standalone/freestanding development, just as
when cross-compiling from a non-`aarch64` platform.
