# `inclusive_range_syntax`

The tracking issue for this feature is: [#28237]

[#28237]: https://github.com/rust-lang/rust/issues/28237

------------------------

To get a range that goes from 0 to 10 and includes the value 10, you
can write `0..=10`:

```rust
#![feature(inclusive_range_syntax)]

fn main() {
    for i in 0..=10 {
        println!("{}", i);
    }
}
```
