# `*-unknown-hyperlight`

**Tier: 3**

These targets are used to build guest executables for the
[Hyperlight](https://github.com/hyperlight-dev/hyperlight) sandboxing
solution.

## Target maintainers

- [@syntactically](https://github.com/syntactically)
- [@yoshuawuyts](https://github.com/yoshuawuyts)

## Requirements

These targets can only be used for cross-compilation.  They should not
require anything special as host tooling, but the usual way to use
them is via the
[cargo-hyperlight](https://github.com/hyperlight-dev/cargo-hyperlight)
tool.

Hyperlight targets do not have any `std` support, but do fully support
`alloc`.

These targets assume ARMv8.1-A (with Neon) on aarch64 and SSE2 on
x86_64, which are required by core Hyperlight libraries.  The targets
are only expected to work with recent versions of Hyperlight.

Hyperlight uses the official hardfloat calling convention of each
architecture for `extern "C"`.

Hyperlight guest binaries use the ELF file format.

## Building the target

These targets can be built by adding them to the `target` list in
`bootstrap.toml`.

```toml
[build]
build-stage = 1
target = [
    "<HOST_TARGET>",
    "<ARCH>-unknown-hyperlight",
]

[target.<ARCH>-unknown-hyperlight]
no-std = true
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To
compile for this target, you will either need to build Rust with the
target enabled (see "Building the target" above), or build your own
copy of `core` by using `build-std` or similar.  When using
`cargo-hyperlight`, the tool will manage this automatically.

## Testing

Binaries built for this target need to use the
[`hyperlight-guest-bin`](https://crates.io/crates/hyperlight-guest-bin)
crate.  A given binary can then be loaded by a host process that uses
[`hyperlight-host`](https://crates.io/crates/hyperlight-host) and
agrees with it on a host/guest interface.  For more details on the
programming model and how to write a host, see those crates and
documentation on the Hyperlight [website](https://hyperlight.org) and
in the [repository](https://github.com/hyperlight-dev/hyperlight).

## Cross-compilation toolchains and C code

This target does support C code through
[cargo-hyperlight](https://github.com/hyperlight-dev/cargo-hyperlight),
which supports wrapping an `<arch>-unknown-none` C toolchain into
something that can build targeting Hyperlight.
