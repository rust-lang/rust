// compile-flags: --crate-type lib --edition 2018

#![crate_name = "foo"]
#![feature(rustdoc_internals)]

pub trait Foo {}

// @has foo/trait.Foo.html
// @has - '//section[@id="impl-Foo-for-(T%2C)"]/h3' 'impl<T> Foo for (T₁, T₂, …, Tₙ)'
#[doc(tuple_variadic)]
impl<T> Foo for (T,) {}

pub trait Bar {}

// @has foo/trait.Bar.html
// @has - '//section[@id="impl-Bar-for-(U%2C)"]/h3' 'impl<U: Foo> Bar for (U₁, U₂, …, Uₙ)'
#[doc(tuple_variadic)]
impl<U: Foo> Bar for (U,) {}
