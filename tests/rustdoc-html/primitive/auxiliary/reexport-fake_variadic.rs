#![feature(rustdoc_internals)]

pub trait Foo {}

#[doc(fake_variadic)]
impl<T> Foo for (T,) {}
