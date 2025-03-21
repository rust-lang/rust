# `*-lynxos_178-*`

**Tier: 3**

Targets for the LynxOS-178 operating system.

[LynxOS-178](https://www.lynx.com/products/lynxos-178-do-178c-certified-posix-rtos)
is a commercial RTOS designed for safety-critical real-time systems.  It is
developed by Lynx Software Technologies as part of the
[MOSA.ic](https://www.lynx.com/solutions/safe-and-secure-operating-environment)
product suite.

Target triplets available:
- `x86_64-unknown-lynxos_178-elf`

## Target maintainers

- Tim Newsome, https://github.com/tnewsome-lynx

## Requirements

To build Rust programs for LynxOS-178, you must first have LYNX MOSA.ic
installed on the build machine.

This target supports only cross-compilation, from the same hosts supported by
the Lynx CDK.

Currently only `no_std` programs are supported. Work to support `std` is in
progress.

## Building the target

You can build Rust with support for your chosen targets by adding them to the
`target` list in `config.toml`.

When executing `cargo`, the `ENV_PREFIX` environment variable must point at the
LynxOS-178 Environment, e.g. `/opt/lynxos178/2025.01.0-a5b49faee-826/x86/dev`.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Binaries built with rust can be provided to a LynxOS-178 instance on its file
system, where they can be executed. Rust binaries tend to be large, so it may
be necessary to strip them first.

It is possible to run the Rust testsuite by providing a test runner that takes
the test binary and executes it under LynxOS-178. Most (all?) tests won't run
without std support though, which is not yet supported.

## Cross-compilation toolchains and C code

LYNX MOSA.ic comes with all the tools required to cross-compile C code for
LynxOS-178.
