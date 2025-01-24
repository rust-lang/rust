# `trait_upcasting`

The tracking issue for this feature is: [#65991]

[#65991]: https://github.com/rust-lang/rust/issues/65991

------------------------

The `trait_upcasting` feature adds support for trait upcasting coercion. This allows a
trait object of type `dyn Bar` to be cast to a trait object of type `dyn Foo`
so long as `Bar: Foo`.

```rust,edition2018
#![feature(trait_upcasting)]

trait Foo {}

trait Bar: Foo {}

impl Foo for i32 {}

impl<T: Foo + ?Sized> Bar for T {}

let bar: &dyn Bar = &123;
let foo: &dyn Foo = bar;
```
