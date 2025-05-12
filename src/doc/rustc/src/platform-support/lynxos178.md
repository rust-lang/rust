# `*-lynxos178-*`

**Tier: 3**

Targets for the LynxOS-178 operating system.

[LynxOS-178](https://www.lynx.com/products/lynxos-178-do-178c-certified-posix-rtos)
is a commercial RTOS designed for safety-critical real-time systems.  It is
developed by Lynx Software Technologies as part of the
[MOSA.ic](https://www.lynx.com/solutions/safe-and-secure-operating-environment)
product suite.

Target triples available:
- `x86_64-lynx-lynxos178`

## Target maintainers

- Renat Fatykhov, https://github.com/rfatykhov-lynx

## Requirements

To build Rust programs for LynxOS-178, you must first have LYNX MOSA.ic
installed on the build machine.

This target supports only cross-compilation, from the same hosts supported by
the Lynx CDK.

Currently only `no_std` programs are supported. Work to support `std` is in
progress.

## Building the target

You can build Rust with support for x86_64-lynx-lynxos178 by adding that
to the `target` list in `config.toml`, and then running `./x build --target
x86_64-lynx-lynxos178 compiler`.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will need to build Rust with the target enabled (see "Building
the target" above).

Before executing `cargo`, you must configure the environment to build LynxOS-178
binaries by running `source setup.sh` from the los178 directory.

If your program/crates contain procedural macros, Rust must be able to build
binaries for the host as well. The host gcc is hidden by sourcing setup.sh.  To
deal with this, add the following to your project's `.cargo/config.toml`:
```toml
[target.x86_64-unknown-linux-gnu]
linker = "lynx-host-gcc"
```
(If necessary substitute your host target triple for x86_64-unknown-linux-gnu.)

To point `cargo` at the correct rustc binary, set the RUSTC environment
variable.

The core library should be usable. You can try by building it as part of your
project:
```bash
cargo +nightly build -Z build-std=core --target x86_64-lynx-lynxos178
```

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
