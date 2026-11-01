# `x86_64-fortanix-unknown-sgx`

**Tier: 2**

Secure enclaves using [Intel Software Guard Extensions
(SGX)](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
based on the ABI defined by Fortanix for the [Enclave Development Platform
(EDP)](https://edp.fortanix.com/).

## Target maintainers

[@jethrogb](https://github.com/jethrogb)
[@raoulstrackx](https://github.com/raoulstrackx)
[@aditijannu](https://github.com/aditijannu)

Further contacts:

The [EDP team](mailto:edp.maintainers@fortanix.com) at Fortanix.

## Requirements

The target supports `std` with a default allocator. Only cross compilation is
supported.

Binaries support all CPUs that include Intel SGX. Only 64-bit mode is supported.

Not all `std` features are supported, see [Using Rust's
std](https://edp.fortanix.com/docs/concepts/rust-std/) for details.

The `extern "C"` calling convention is the System V AMD64 ABI.

The supported ABI is the
[fortanix-sgx-abi](https://edp.fortanix.com/docs/api/fortanix_sgx_abi/index.html).

The compiler output is ELF, but the native format for the platform is the SGX
stream (SGXS) format. A converter like
[ftxsgx-elf2sgxs](https://crates.io/crates/fortanix-sgx-tools) is needed.

Programs in SGXS format adhering to the Fortanix SGX ABI can be run with any
compatible runner, such as
[ftxsgx-runner](https://crates.io/crates/fortanix-sgx-tools).

See the [EDP installation
guide](https://edp.fortanix.com/docs/installation/guide/) for recommendations
on how to setup a development and runtime environment.

## Building the target

As a tier 2 target, the target is built by the Rust project.

You can configure bootstrap like so:

```toml
[build]
build-stage = 1
target = ["x86_64-fortanix-unknown-sgx"]
```

## Building Rust programs

Standard build flows using `cargo` or `rustc` should work.

## Testing

The Rust test suite as well as custom unit and integration tests will run on
hardware that has Intel SGX enabled if a cargo runner is configured correctly,
see the requirements section.

## Cross-compilation toolchains and C code

C code is not generally supported, as there is no libc. C code compiled for
x86-64 in freestanding mode using the System V AMD64 ABI may work. The
[rs-libc](https://crates.io/crates/rs-libc) crate contains a subset of libc
that's known to work with this target.
