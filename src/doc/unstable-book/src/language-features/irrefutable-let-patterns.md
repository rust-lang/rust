# `irrefutable_let_patterns`

The tracking issue for this feature is: [#44495]

[#44495]: https://github.com/rust-lang/rust/issues/44495

------------------------

This feature changes the way that "irrefutable patterns" are handled
in the `if let` and `while let` forms. An *irrefutable pattern* is one
that cannot fail to match -- for example, the `_` pattern matches any
value, and hence it is "irrefutable". Without this feature, using an
irrefutable pattern in an `if let` gives a hard error (since often
this indicates programmer error). But when the feature is enabled, the
error becomes a lint (since in some cases irrefutable patterns are
expected). This means you can use `#[allow]` to silence the lint:

```rust
#![feature(irrefutable_let_patterns)]

#[allow(irrefutable_let_patterns)]
fn main() {
    // These two examples used to be errors, but now they
    // trigger a lint (that is allowed):
    if let _ = 5 {}
    while let _ = 5 {}
}
```
