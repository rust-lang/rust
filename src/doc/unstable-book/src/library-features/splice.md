# `splice`

The tracking issue for this feature is: [#44643]

[#44643]: https://github.com/rust-lang/rust/issues/44643

------------------------

The `splice()` method on `String` allows you to replace a range
of values in a string with another range of values.

A simple example:

```rust
#![feature(splice)]
let mut s = String::from("α is alpha, β is beta");
let beta_offset = s.find('β').unwrap_or(s.len());

// Replace the range up until the β from the string
s.splice(..beta_offset, "Α is capital alpha; ");
assert_eq!(s, "Α is capital alpha; β is beta");
```
