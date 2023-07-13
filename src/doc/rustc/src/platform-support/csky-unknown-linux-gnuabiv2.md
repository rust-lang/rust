# `csky-unknown-linux-gnuabiv2`

**Tier: 3**

This target supports [C-SKY](https://github.com/c-sky) v2 CPUs with `glibc`.

https://c-sky.github.io/
## Target maintainers

* [@Dirreke](https://github.com/Dirreke)

## Requirements


## Building the target

add `csky-unknown-linux-gnuabiv2` to the `target` list in `config.toml` and `./x build`.

## Building Rust programs

Rust programs can be built for that target:

```text
cargo +stage2 --target csky-unknown-linux-gnuabiv2 your-code.rs
```

## Testing

Currently there is no support to run the rustc test suite for this target.

## Cross-compilation toolchains and C code

This target can be cross-compiled from `x86_64` on either Linux systems with [`csky-linux-gunabiv2-tools-x86_64-glibc-linux`](https://github.com/c-sky/toolchain-build).
