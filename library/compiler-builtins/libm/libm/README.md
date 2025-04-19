# `libm`

A port of [MUSL]'s libm to Rust.

[MUSL]: https://musl.libc.org/

## Goals

The short term goal of this library is to [enable math support (e.g. `sin`, `atan2`) for the
`wasm32-unknown-unknown` target][wasm] (cf. [rust-lang/compiler-builtins][pr]). The longer
term goal is to enable [math support in the `core` crate][core].

[wasm]: https://github.com/rust-lang/libm/milestone/1
[pr]: https://github.com/rust-lang/compiler-builtins/pull/248
[core]: https://github.com/rust-lang/libm/milestone/2

## Already usable

This crate is [on crates.io] and can be used today in stable `#![no_std]` programs.

The API documentation can be found [here](https://docs.rs/libm).

[on crates.io]: https://crates.io/crates/libm

## Benchmark
[benchmark]: #benchmark

The benchmarks are located in `crates/libm-bench` and require a nightly Rust toolchain.
To run all benchmarks:

> cargo +nightly bench --all

## Contributing

Please check [CONTRIBUTING.md](CONTRIBUTING.md)

## Minimum Rust version policy

This crate supports rustc 1.63 and newer.

## License

Usage is licensed under the MIT license ([LICENSE-MIT](LICENSE-MIT) or
https://opensource.org/licenses/MIT).


### Contribution

Contributions are licensed under both the MIT license and the Apache License,
Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
https://www.apache.org/licenses/LICENSE-2.0). Unless you explicitly state
otherwise, any contribution intentionally submitted for inclusion in the work
by you, as defined in the Apache-2.0 license, shall be dual licensed as
mentioned, without any additional terms or conditions.

See `LICENSE.txt` for full details.
