//@ revisions: any_lt static_lt
//@[static_lt] check-pass

#![feature(negative_impls)]
#![feature(with_negative_coherence)]

trait Foo {}

impl<T> !Foo for &'static T {}

trait Bar {}

impl<T> Bar for T where T: Foo {}

#[cfg(any_lt)]
impl<T> Bar for &T {}
//[any_lt]~^ ERROR conflicting implementations of trait `Bar` for type `&_`

#[cfg(static_lt)]
impl<T> Bar for &'static T {}


fn main() {}
