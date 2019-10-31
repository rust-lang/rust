# `trait_upcasting`

The tracking issue for this feature is: [#31436]

[#65991]: https://github.com/rust-lang/rust/issues/65991

------------------------

The `trait_upcasting` feature adds support for trait upcasting. This allows a
trait object of type `dyn Foo` to be cast to a trait object of type `dyn Bar`
so long as `Foo: Bar`.

```rust,edition2018
#![feature(trait_upcasting)]

trait Foo {}

trait Bar: Foo {}

impl Foo for i32 {}

impl<T: Foo + ?Sized> Bar for T {}

let foo: &dyn Foo = &123;
let bar: &dyn Bar = foo;
```
