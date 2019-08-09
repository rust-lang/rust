# `box_syntax`

The tracking issue for this feature is: [#49733]

[#49733]: https://github.com/rust-lang/rust/issues/49733

See also [`box_patterns`](box-patterns.md)

------------------------

Currently the only stable way to create a `Box` is via the `Box::new` method.
Also it is not possible in stable Rust to destructure a `Box` in a match
pattern. The unstable `box` keyword can be used to create a `Box`. An example
usage would be:

```rust
#![feature(box_syntax)]

fn main() {
    let b = box 5;
}
```
