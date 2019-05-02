# How to contribute

- Pick your favorite math function from the [issue tracker].
- Look for the C implementation of the function in the [MUSL source code][src].
- Copy paste the C code into a Rust file in the `src/math` directory and adjust
  `src/math/mod.rs` accordingly. Also, uncomment the corresponding trait method
  in `src/lib.rs`.
- Write some simple tests in your module (using `#[test]`)
- Run `cargo test` to make sure it works
- Run `cargo test --features musl-reference-tests` to compare your
  implementation against musl's
- Send us a pull request! Make sure to run `cargo fmt` on your code before
  sending the PR. Also include "closes #42" in the PR description to close the
  corresponding issue.
- :tada:

[issue tracker]: https://github.com/rust-lang-nursery/libm/issues
[src]: https://git.musl-libc.org/cgit/musl/tree/src/math
[`src/math/truncf.rs`]: https://github.com/rust-lang-nursery/libm/blob/master/src/math/truncf.rs

Check [PR #65] for an example.

[PR #65]: https://github.com/rust-lang-nursery/libm/pull/65

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
  point literals. Rust (the language) doesn't support these kind of literals. The best way I have
  found to deal with these literals is to turn them into their integer representation using the
  [`hexf!`] macro and then turn them back into floats. See below:

[`hexf!`]: https://crates.io/crates/hexf

``` rust
// Step 1: write a program to convert the float into its integer representation
#[macro_use]
extern crate hexf;

fn main() {
    println!("{:#x}", hexf32!("0x1.0p127").to_bits());
}
```

``` console
$ # Step 2: run the program
$ cargo run
0x7f000000
```

``` rust
// Step 3: copy paste the output into libm
let x1p127 = f32::from_bits(0x7f000000); // 0x1p127f === 2 ^ 12
```

- Rust code panics on arithmetic overflows when not optimized. You may need to use the [`Wrapping`]
  newtype to avoid this problem.

[`Wrapping`]: https://doc.rust-lang.org/std/num/struct.Wrapping.html

## Testing

Normal tests can be executed with:

```
cargo test
```

If you'd like to run tests with randomized inputs that get compared against musl
itself, you'll need to be on a Linux system and then you can execute:

```
cargo test --features musl-reference-tests
```

Note that you may need to pass `--release` to Cargo if there are errors related
to integer overflow.
