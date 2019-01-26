# `crate_visibility_modifier`

The tracking issue for this feature is: [#53120]

[#53120]: https://github.com/rust-lang/rust/issues/53120

-----

The `crate_visibility_modifier` feature allows the `crate` keyword to be used
as a visibility modifier synonymous to `pub(crate)`, indicating that a type
(function, _&c._) is to be visible to the entire enclosing crate, but not to
other crates.

```rust
#![feature(crate_visibility_modifier)]

crate struct Foo {
    bar: usize,
}
```
