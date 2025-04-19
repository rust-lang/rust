# How to contribute

- Pick your favorite math function from the [issue tracker].
- Look for the C implementation of the function in the [MUSL source code][src].
- Copy paste the C code into a Rust file in the `src/math` directory and adjust
  `src/math/mod.rs` accordingly. Also, uncomment the corresponding trait method
  in `src/lib.rs`.
- Write some simple tests in your module (using `#[test]`)
- Run `cargo test` to make sure it works. Full tests are only run when enabling
  features, see [Testing](#testing) below.
- Send us a pull request! Make sure to run `cargo fmt` on your code before
  sending the PR. Also include "closes #42" in the PR description to close the
  corresponding issue.
- :tada:

[issue tracker]: https://github.com/rust-lang/libm/issues
[src]: https://git.musl-libc.org/cgit/musl/tree/src/math
[`src/math/truncf.rs`]: https://github.com/rust-lang/libm/blob/master/src/math/truncf.rs

Check [PR #65] for an example.

[PR #65]: https://github.com/rust-lang/libm/pull/65

## Tips and tricks

- *IMPORTANT* The code in this crate will end up being used in the `core` crate so it can **not**
  have any external dependencies (other than `core` itself).

- Only use relative imports within the `math` directory / module, e.g. `use self::fabs::fabs` or
`use super::k_cos`. Absolute imports from core are OK, e.g. `use core::u64`.

- To reinterpret a float as an integer use the `to_bits` method. The MUSL code uses the
  `GET_FLOAT_WORD` macro, or a union, to do this operation.

- To reinterpret an integer as a float use the `f32::from_bits` constructor. The MUSL code uses the
  `SET_FLOAT_WORD` macro, or a union, to do this operation.

- You may use other methods from core like `f64::is_nan`, etc. as appropriate.

- If you're implementing one of the private double-underscore functions, take a look at the
  "source" name in the comment at the top for an idea for alternate naming. For example, `__sin`
  was renamed to `k_sin` after the FreeBSD source code naming. Do `use` these private functions in
  `mod.rs`.

- You may encounter weird literals like `0x1p127f` in the MUSL code. These are hexadecimal floating
  point literals. Rust (the language) doesn't support these kind of literals. This crate provides
  two macros, `hf32!` and `hf64!`, which convert string literals to floats at compile time.

  ```rust
  assert_eq!(hf32!("0x1.ffep+8").to_bits(), 0x43fff000);
  assert_eq!(hf64!("0x1.ffep+8").to_bits(), 0x407ffe0000000000);
  ```

- Rust code panics on arithmetic overflows when not optimized. You may need to use the [`Wrapping`]
  newtype to avoid this problem, or individual methods like [`wrapping_add`].

[`Wrapping`]: https://doc.rust-lang.org/std/num/struct.Wrapping.html
[`wrapping_add`]: https://doc.rust-lang.org/std/primitive.u32.html#method.wrapping_add

## Testing

Normal tests can be executed with:

```sh
# Tests against musl require that the submodule is up to date.
git submodule init
git submodule update

# `--release` ables more test cases
cargo test --release
```

If you are on a system that cannot build musl or MPFR, passing
`--no-default-features` will run some limited tests.

The multiprecision tests use the [`rug`] crate for bindings to MPFR. MPFR can
be difficult to build on non-Unix systems, refer to [`gmp_mpfr_sys`] for help.

`build-musl` does not build with MSVC, Wasm, or Thumb.

[`rug`]: https://docs.rs/rug/latest/rug/
[`gmp_mpfr_sys`]: https://docs.rs/gmp-mpfr-sys/1.6.4/gmp_mpfr_sys/
