#![feature(rustdoc_internals)]

trait Mine {}

// This one is fine
#[doc(tuple_variadic)]
impl<T> Mine for (T,) {}

trait Mine2 {}

// This one is not
#[doc(tuple_variadic)] //~ ERROR
impl<T, U> Mine for (T,U) {}

fn main() {}
