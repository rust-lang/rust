# `libm`

A port of [MUSL]'s libm to Rust.

[MUSL]: https://www.musl-libc.org/

## Goals

The short term goal of this library is to [enable math support (e.g. `sin`, `atan2`) for the
`wasm32-unknown-unknown` target][wasm] (cf. [rust-lang-nursery/compiler-builtins][pr]). The longer
term goal is to enable [math support in the `core` crate][core].

[wasm]: https://github.com/japaric/libm/milestone/1
[pr]: https://github.com/rust-lang-nursery/compiler-builtins/pull/248
[core]: https://github.com/japaric/libm/milestone/2

## Contributing

Please check [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
