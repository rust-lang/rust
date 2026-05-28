# aarch64-nintendo-switch-freestanding

**Tier: 3**

Nintendo Switch with pure-Rust toolchain.

## Target Maintainers

[@leo60228](https://github.com/leo60228)
[@jam1garner](https://github.com/jam1garner)

## Requirements

This target is cross-compiled.
It has no special requirements for the host.

## Building

The target can be built by enabling it for a `rustc` build:

```toml
[build]
build-stage = 1
target = ["aarch64-nintendo-switch-freestanding"]
```

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

Currently there is no support to run the rustc test suite for this target.

## Building Rust programs

If `rustc` has support for that target and the library artifacts are available,
then Rust programs can be built for that target:

```text
rustc --target aarch64-nintendo-switch-freestanding your-code.rs
```

To generate binaries in the NRO format that can be easily run on-device, you
can use [cargo-nx](https://github.com/aarch64-switch-rs/cargo-nx):

```text
cargo nx --triple=aarch64-nintendo-switch-freestanding
```
