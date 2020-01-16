# `member_constraints`

The tracking issue for this feature is: [#61997]

[#61997]: https://github.com/rust-lang/rust/issues/61997

------------------------

The `member_constraints` feature gate lets you use `impl Trait` syntax with
multiple unrelated lifetime parameters.

A simple example is:

```rust
#![feature(member_constraints)]

trait Trait<'a, 'b> { }
impl<T> Trait<'_, '_> for T {}

fn foo<'a, 'b>(x: &'a u32, y: &'b u32) -> impl Trait<'a, 'b> {
  (x, y)
}

fn main() { }
```

Without the `member_constraints` feature gate, the above example is an
error because both `'a` and `'b` appear in the impl Trait bounds, but
neither outlives the other.
