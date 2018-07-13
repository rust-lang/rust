# How to contribute

- Pick your favorite math function from the [issue tracker].
- Look for the C implementation of the function in the [MUSL source code][src].
- Copy paste the C code into a Rust file in the `src/math` directory and adjust `src/math/mod.rs`
  accordingly. Also, uncomment the corresponding trait method in `src/lib.rs`.
- Run `cargo watch check` and fix the compiler errors.
- Tweak the bottom of `test-generator/src/main.rs` to add your function to the test suite.
- If you can, run the full test suite locally (see the [testing](#testing) section below). If you
  can't, no problem! Your PR will be fully tested automatically. Though you may still want to add
  and run some unit tests. See the bottom of [`src/math/truncf.rs`] for an example of such tests;
  you can run unit tests with the `cargo test --lib` command.
- Send us a pull request!
- :tada:

[issue tracker]: https://github.com/japaric/libm/issues
[src]: https://git.musl-libc.org/cgit/musl/tree/src/math
[`src/math/truncf.rs`]: https://github.com/japaric/libm/blob/master/src/math/truncf.rs

Check [PR #65] for an example.

[PR #65]: https://github.com/japaric/libm/pull/65

## Tips and tricks

- *IMPORTANT* The code in this crate will end up being used in the `core` crate so it can **not**
  have any external dependencies (other than `core` itself).

- Only use relative imports within the `math` directory / module, e.g. `use self::fabs::fabs` or
`use super::isnanf`. Absolute imports from core are OK, e.g. `use core::u64`.

- To reinterpret a float as an integer use the `to_bits` method. The MUSL code uses the
  `GET_FLOAT_WORD` macro, or a union, to do this operation.

- To reinterpret an integer as a float use the `f32::from_bits` constructor. The MUSL code uses the
  `SET_FLOAT_WORD` macro, or a union, to do this operation.

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

The test suite of this crate can only be run on x86_64 Linux systems using the following commands:

``` console
$ # The test suite depends on the `cross` tool so install it if you don't have it
$ cargo install cross

$ # and the `cross` tool requires docker to be running
$ systemctl start docker

$ # execute the test suite for the x86_64 target
$ TARGET=x86_64-unknown-linux-gnu bash ci/script.sh

$ # execute the test suite for the ARMv7 target
$ TARGET=armv7-unknown-linux-gnueabihf bash ci/script.sh
```
