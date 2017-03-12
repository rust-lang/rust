# `i128_type`

The tracking issue for this feature is: [#35118]

[#35118]: https://github.com/rust-lang/rust/issues/35118

------------------------

The `i128_type` feature adds support for 128 bit signed and unsigned integer
types.

```rust
#![feature(i128_type)]

fn main() {
    assert_eq!(1u128 + 1u128, 2u128);
    assert_eq!(u128::min_value(), 0);
    assert_eq!(u128::max_value(), 340282366920938463463374607431768211455);

    assert_eq!(1i128 - 2i128, -1i128);
    assert_eq!(i128::min_value(), -170141183460469231731687303715884105728);
    assert_eq!(i128::max_value(), 170141183460469231731687303715884105727);
}
```

