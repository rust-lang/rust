# `*-l4re-uclibc`

**Tier: 3**

[L4Re] is an open source, microkernel-based operating system and hypervisor.

Target triplets available so far:

- x86_64-unknown-l4re-uclibc
- aarch64-unknown-l4re-uclibc

## Target maintainers

- Marius Melzer ([@farao](https://github.com/farao))

## Requirements

The L4Re targets are cross-compiled from a host environment, commonly Linux.
See [Getting Started] for options to set up L4Re.

The L4Re sources can be found in the [Github Repos].

## Building an L4Re Rust Toolchain

Configure one or several of the above L4Re targets and also add the host triple
in config.toml and build Rust as documented. Start off the toolchain by copying
`build/host/stage2/` to a self-chosen location.

For each target, build an L4Re sysroot directory by running `make sysroot` in
the L4Re build directory. Copy the content of `sysroot/usr/lib/` into the
`self-contained` directory of the respective target in the Rust Toolchain
directory tree.

Use rustup to install the L4Re Rust Toolchain locally:

```sh
rustup toolchain link l4re <path-to-toolchain>
```

Now use the toolchain via a cargo (or directly a rustc) installed via `rustup`:

```sh
cargo +l4re build --target <l4re-triple>
```

or

```sh
rustc +l4re --target <l4re-triple> <rs-file>
```

## Run Rust Programs on L4Re

You can run an L4Re application written in Rust just like any other externally
built (meaning not build with the L4Re build system) L4Re binary. A good option
is to build an L4Re image and add the application binary to the image and run it
via the ned script. The image can then be put on hardware or run on Qemu.

See [l4re.org](https://l4re.org) for more information.

[L4Re]: https://l4re.org
[Getting Started]: https://l4re.org/getting_started
[Github Repos]: https://github.com/L4Re
