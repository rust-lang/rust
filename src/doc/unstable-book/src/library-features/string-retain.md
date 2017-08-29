# `string_retain`

The tracking issue for this feature is: [#43874]

[#43874]: https://github.com/rust-lang/rust/issues/43874

------------------------

Retains only the characters specified by the predicate.

In other words, remove all characters `c` such that `f(c)` returns `false`.
This method operates in place and preserves the order of the retained
characters.

```rust
#![feature(string_retain)]

let mut s = String::from("f_o_ob_ar");

s.retain(|c| c != '_');

assert_eq!(s, "foobar");
```
