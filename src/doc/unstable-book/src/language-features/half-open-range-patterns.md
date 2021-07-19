# `half_open_range_patterns`

The tracking issue for this feature is: [#67264]
It is part of the `#![exclusive_range_pattern]` feature,
tracked at [#37854].

[#67264]: https://github.com/rust-lang/rust/issues/67264
[#37854]: https://github.com/rust-lang/rust/issues/37854
-----

The `half_open_range_patterns` feature allows RangeTo patterns
(`..10`) to be used in appropriate pattern matching contexts.
This requires also enabling the `exclusive_range_pattern` feature.

It also enabled RangeFrom patterns but that has since been
stabilized.

```rust
#![feature(half_open_range_patterns)]
#![feature(exclusive_range_pattern)]
    let x = 5;
    match x {
        ..0 => println!("negative!"), // "RangeTo" pattern. Unstable.
        0 => println!("zero!"),
        1.. => println!("positive!"), // "RangeFrom" pattern. Stable.
    }
```
