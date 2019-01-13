# `const_fn`

The tracking issue for this feature is: [#57563]

[#57563]: https://github.com/rust-lang/rust/issues/57563

------------------------

The `const_fn` feature allows marking free functions and inherent methods as
`const`, enabling them to be called in constants contexts, with constant
arguments.

## Examples

```rust
#![feature(const_fn)]

const fn double(x: i32) -> i32 {
    x * 2
}

const FIVE: i32 = 5;
const TEN: i32 = double(FIVE);

fn main() {
    assert_eq!(5, FIVE);
    assert_eq!(10, TEN);
}
```
