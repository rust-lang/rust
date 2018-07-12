# `libm`

A port of [MUSL]'s libm to Rust.

[MUSL]: https://www.musl-libc.org/

## Testing

The test suite of this crate can only be run on x86_64 Linux systems.

```
$ # The test suite depends on the `cross` tool so install it if you don't have it
$ cargo install cross

$ # and the `cross` tool requires docker to be running
$ systemctl start docker

$ # execute the test suite for the x86_64 target
$ TARGET=x86_64-unknown-linux-gnu bash ci/script.sh

$ # execute the test suite for the ARMv7 target
$ TARGET=armv7-unknown-linux-gnueabihf bash ci/script.sh
```

## Contributing

- Pick your favorite math function from the list below.
- Look for the C implementation of the function in the [MUSL source code][src].
- Copy paste the C code into a Rust file in the `src` directory and adjust `src/lib.rs` accordingly.
- Run `cargo watch check` and fix the compiler errors.
- If you can, run the test suite locally. If you can't, no problem! Your PR will be tested
  automatically.
- Send us a pull request!
- :tada:

### Notes

- To reinterpret a float as an integer use the `to_bits` method. The MUSL code uses the
  `GET_FLOAT_WORD` macro, or a union, to do this operation.

- To reinterpret an integer as a float use the `f32::from_bits` constructor. The MUSL code uses the
  `SET_FLOAT_WORD` macro, or a union, to do this operation.

- Rust code panics on arithmetic overflows when not optimized. You may need to use the [`Wrapping`]
  newtype to avoid this problem.

[src]: https://git.musl-libc.org/cgit/musl/tree/src/math
[`Wrapping`]: https://doc.rust-lang.org/std/num/struct.Wrapping.html

## Progress

### Functions wanted by the wasm WG

cf. [rustwasm/team#84](https://github.com/rustwasm/team/issues/84).

- [ ] acos
- [ ] asin
- [ ] atan
- [ ] atan2
- [ ] cbrt
- [ ] cos
- [ ] cosf
- [ ] cosh
- [ ] exp
- [ ] exp2
- [ ] exp2f
- [ ] expf
- [ ] expm1
- [ ] fma
- [ ] fmaf
- [ ] fmod
- [ ] fmodf
- [ ] hypot
- [ ] log
- [ ] log10
- [ ] log10f
- [ ] log1p
- [ ] log2
- [ ] log2f
- [ ] logf
- [ ] pow
- [x] powf
- [ ] round
- [ ] roundf
- [ ] sin
- [ ] sinf
- [ ] sinh
- [ ] tan
- [ ] tanh

### Other functions

- [x] fabsf
- [x] scalbnf
- [x] sqrtf

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
