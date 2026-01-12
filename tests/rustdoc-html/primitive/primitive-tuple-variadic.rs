//@ compile-flags: --crate-type lib
//@ edition: 2018

#![crate_name = "foo"]
#![feature(rustdoc_internals)]

pub trait Foo {}

//@ has foo/trait.Foo.html
//@ has - '//section[@id="impl-Foo-for-(T,)"]/h3' 'impl<T> Foo for (T₁, T₂, …, Tₙ)'
#[doc(fake_variadic)]
impl<T> Foo for (T,) {}

pub trait Bar {}

//@ has foo/trait.Bar.html
//@ has - '//section[@id="impl-Bar-for-(U,)"]/h3' 'impl<U: Foo> Bar for (U₁, U₂, …, Uₙ)'
#[doc(fake_variadic)]
impl<U: Foo> Bar for (U,) {}

pub trait Baz<T> { fn baz(&self) -> T { todo!() } }

//@ has foo/trait.Baz.html
//@ has - '//section[@id="impl-Baz%3C(T,)%3E-for-%5BT;+1%5D"]/h3' 'impl<T> Baz<(T₁, T₂, …, Tₙ)> for [T; N]'
#[doc(fake_variadic)]
impl<T> Baz<(T,)> for [T; 1] {}

//@ has foo/trait.Baz.html
//@ has - '//section[@id="impl-Baz%3C%5BT;+1%5D%3E-for-(T,)"]/h3' 'impl<T> Baz<[T; N]> for (T₁, T₂, …, Tₙ)'
#[doc(fake_variadic)]
impl<T> Baz<[T; 1]> for (T,) {}

//@ has foo/trait.Baz.html
//@ has - '//section[@id="impl-Baz%3CT%3E-for-(T,)"]/h3' 'impl<T> Baz<T> for (T₁, T₂, …, Tₙ)'
#[doc(fake_variadic)]
impl<T> Baz<T> for (T,) {}

pub trait Qux {}

pub struct NewType<T>(T);

//@ has foo/trait.Qux.html
//@ has - '//section[@id="impl-Qux-for-NewType%3C(T,)%3E"]/h3' 'impl<T> Qux for NewType<(T₁, T₂, …, Tₙ)>'
#[doc(fake_variadic)]
impl<T> Qux for NewType<(T,)> {}

//@ has foo/trait.Qux.html
//@ has - '//section[@id="impl-Qux-for-NewType%3CNewType%3C(T,)%3E%3E"]/h3' 'impl<T> Qux for NewType<NewType<(T₁, T₂, …, Tₙ)>>'
#[doc(fake_variadic)]
impl<T> Qux for NewType<NewType<(T,)>> {}

//@ has foo/trait.Qux.html
//@ has - '//section[@id="impl-Qux-for-NewType%3Cfn(T)+-%3E+Out%3E"]/h3' 'impl<T, Out> Qux for NewType<fn(T₁, T₂, …, Tₙ) -> Out>'
#[doc(fake_variadic)]
impl<T, Out> Qux for NewType<fn(T) -> Out> {}
