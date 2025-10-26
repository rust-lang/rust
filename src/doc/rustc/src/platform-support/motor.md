# `x86_64-unknown-motor`

**Tier: 3**

[Motor OS](https://github.com/moturus/motor-os) is a new operating system
for virtualized environments.

## Target maintainers

[@lasiotus](https://github.com/lasiotus)

## Requirements

This target is cross-compiled. There are no special requirements for the host.

Motor OS uses the ELF file format.

## Building the target toolchain

Motor OS target toolchain can be
[built using `x.py`](https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html):

The bootstrap file:

```toml
[build]
host = ["x86_64-unknown-linux-gnu"]
target = ["x86_64-unknown-linux-gnu", "x86_64-unknown-motor"]
```

The build command:

```sh
./x.py build --stage 2 clippy library
```

## Building Rust programs

See the [Hello Motor OS](https://github.com/moturus/motor-os/blob/main/docs/recipes/hello-motor-os.md)
example.

## Testing

Cross-compiled Rust binaries and test artifacts can be executed in Motor OS VMs,
as described in the [build doc](https://github.com/moturus/motor-os/blob/main/docs/build.md)
and the
[Hello Motor OS](https://github.com/moturus/motor-os/blob/main/docs/recipes/hello-motor-os.md)
example.

## Cross-compilation toolchains and C code

C code can be compiled as part of Rust cargo projects. However, there is
no libc support.
