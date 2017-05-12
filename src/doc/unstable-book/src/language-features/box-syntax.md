# `box_syntax`

The tracking issue for this feature is: [#27779]

[#27779]: https://github.com/rust-lang/rust/issues/27779

See also [`box_patterns`](box-patterns.html)

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
