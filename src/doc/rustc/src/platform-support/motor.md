# `x86_64-unknown-motor`

**Tier: 3**

[Motor OS](https://github.com/moturus/motor-os) is a new operating system
for virtualized environments.

## Target maintainers

[@lasiotus](https://github.com/lasiotus)

## Requirements

This target is cross-compiled. There are no special requirements for the host.

Motor OS uses the ELF file format.

## Building the target

The target can be built by enabling it for a `rustc` build, for example:

```toml
[build]
build-stage = 2
target = ["x86_64-unknown-motor"]
```

## Building Rust programs

Rust standard library is fully supported/implemented, but is not yet part of
the official Rust repo, so an out-of-tree building process should be
followed, as described in the
[build doc](https://github.com/moturus/motor-os/blob/main/docs/build.md).

## Testing

Cross-compiled Rust binaries and test artifacts can be executed in Motor OS VMs,
as described in e.g.
[Hello Motor OS](https://github.com/moturus/motor-os/blob/main/docs/recipes/hello-motor-os.md)
example.

## Cross-compilation toolchains and C code

C code can be compiled as part of Rust cargo projects. However, there is
no libc support.
