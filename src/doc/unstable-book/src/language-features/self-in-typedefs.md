# `self_in_typedefs`

The tracking issue for this feature is: [#49303]

[#49303]: https://github.com/rust-lang/rust/issues/49303

------------------------

The `self_in_typedefs` feature gate lets you use the special `Self` identifier
in `struct`, `enum`, and `union` type definitions.

A simple example is:

```rust
#![feature(self_in_typedefs)]

enum List<T>
where
    Self: PartialOrd<Self> // can write `Self` instead of `List<T>`
{
    Nil,
    Cons(T, Box<Self>) // likewise here
}
```
