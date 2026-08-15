# `x86_64-unknown-linux-gnutsan`

**Tier: 2**

Target mirroring `x86_64-unknown-linux-gnu` with ThreadSanitizer enabled by
default.
The goal of this target is to allow shipping TSAN-instrumented standard
libraries through rustup, enabling a fully instrumented binary without requiring
nightly features (build-std).
Once build-std stabilizes, this target is no longer needed and will be removed.

## Target maintainers

- [@jakos-sec](https://github.com/jakos-sec)
- [@1c3t3a](https://github.com/1c3t3a)
- [@rust-lang/project-exploit-mitigations][project-exploit-mitigations]

## Requirements

The target is for cross-compilation only. Host tools are not supported, since
there is no need to have the host tools instrumented with TSAN. std is fully
supported.

In all other aspects the target is equivalent to `x86_64-unknown-linux-gnu`.

## Building the target

The target can be built by enabling it for a rustc build:

```toml
[build]
target = ["x86_64-unknown-linux-gnutsan"]
```

## Building Rust programs

Rust programs can be compiled by adding this target via rustup:

```sh
$ rustup target add x86_64-unknown-linux-gnutsan
```

and then compiling with the target:

```sh
$ rustc foo.rs --target x86_64-unknown-linux-gnutsan
```

## Testing

Created binaries will run on Linux without any external requirements.

## Cross-compilation toolchains and C code

The target supports C code and should use the same toolchain target as
`x86_64-unknown-linux-gnu`.
