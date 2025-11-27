# `x86_64-asan-linux-gnu`

**Tier: 3**

Target mirroring `x86_64-unknown-linux-gnu` with AddressSanitizer enabled by
default.
The goal of this target is to allow shipping ASAN-instrumented standard
libraries through rustup, enabling a fully instrumented binary without requiring
nightly features (build-std).
Once build-std stabilizes, we will likely no longer need this target.

## Target maintainers

- [@jakos-sec](https://github.com/jakos-sec)
- [@1c3t3a](https://github.com/1c3t3a)
- [@rust-lang/project-exploit-mitigations][project-exploit-mitigations]

## Requirements

The target is for cross-compilation only. Host tools are not supported, since
there is no need to have the host tools instrumented with ASAN. std is fully
supported.

In all other aspects the target is equivalent to `x86_64-unknown-linux-gnu`.

## Building the target

The target can be built by enabling it for a rustc build:

```toml
[build]
target = ["x86_64-asan-linux-gnu"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

Compilation can be done with:

```text
rustc --target x86_64-asan-linux-gnu your-code.rs
```

## Testing

Created binaries will run on Linux without any external requirements.

## Cross-compilation toolchains and C code

The target supports C code and should use the same toolchain target as
`x86_64-unknown-linux-gnu`.
