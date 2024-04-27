# `exclusive_range_pattern`

The tracking issue for this feature is: [#37854].


[#67264]: https://github.com/rust-lang/rust/issues/67264
[#37854]: https://github.com/rust-lang/rust/issues/37854
-----

The `exclusive_range_pattern` feature allows non-inclusive range
patterns (`0..10`) to be used in appropriate pattern matching
contexts. It also can be combined with `#![feature(half_open_range_patterns]`
to be able to use RangeTo patterns (`..10`).

It also enabled RangeFrom patterns but that has since been
stabilized.

```rust
#![feature(exclusive_range_pattern)]
    let x = 5;
    match x {
        0..10 => println!("single digit"),
        10 => println!("ten isn't part of the above range"),
        _ => println!("nor is everything else.")
    }
```
