# `inline_const_pat`

The tracking issue for this feature is: [#76001]

See also [`inline_const`](inline-const.md)

------

This feature allows you to use inline constant expressions in pattern position:

```rust
#![feature(inline_const_pat)]

const fn one() -> i32 { 1 }

let some_int = 3;
match some_int {
    const { 1 + 2 } => println!("Matched 1 + 2"),
    const { one() } => println!("Matched const fn returning 1"),
    _ => println!("Didn't match anything :("),
}
```

[#76001]: https://github.com/rust-lang/rust/issues/76001
