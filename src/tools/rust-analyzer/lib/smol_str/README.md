# smol_str

[![CI](https://github.com/rust-analyzer/smol_str/workflows/CI/badge.svg)](https://github.com/rust-analyzer/smol_str/actions?query=branch%3Amaster+workflow%3ACI)
[![Crates.io](https://img.shields.io/crates/v/smol_str.svg)](https://crates.io/crates/smol_str)
[![API reference](https://docs.rs/smol_str/badge.svg)](https://docs.rs/smol_str/)


A `SmolStr` is a string type that has the following properties:

* `size_of::<SmolStr>() == 24` (therefore `== size_of::<String>()` on 64 bit platforms)
* `Clone` is `O(1)`
* Strings are stack-allocated if they are:
    * Up to 23 bytes long
    * Longer than 23 bytes, but substrings of `WS` (see `src/lib.rs`). Such strings consist
    solely of consecutive newlines, followed by consecutive spaces
* If a string does not satisfy the aforementioned conditions, it is heap-allocated
* Additionally, a `SmolStr` can be explicitly created from a `&'static str` without allocation

Unlike `String`, however, `SmolStr` is immutable. The primary use case for
`SmolStr` is a good enough default storage for tokens of typical programming
languages. Strings consisting of a series of newlines, followed by a series of
whitespace are a typical pattern in computer programs because of indentation.
Note that a specialized interner might be a better solution for some use cases.

## Benchmarks
Run criterion benches with
```sh
cargo bench --bench \* -- --quick
```

## MSRV Policy

Minimal Supported Rust Version: latest stable.

Bumping MSRV is not considered a semver-breaking change.
