# `splice`

The tracking issue for this feature is: [#32310]

[#32310]: https://github.com/rust-lang/rust/issues/32310

------------------------

The `splice()` method on `Vec` and `String` allows you to replace a range
of values in a vector or string with another range of values, and returns
the replaced values.

A simple example:

```rust
#![feature(splice)]
let mut s = String::from("α is alpha, β is beta");
let beta_offset = s.find('β').unwrap_or(s.len());

// Replace the range up until the β from the string
let t: String = s.splice(..beta_offset, "Α is capital alpha; ").collect();
assert_eq!(t, "α is alpha, ");
assert_eq!(s, "Α is capital alpha; β is beta");
```