#![feature(coerce_unsized)]

use std::any::Any;
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    data: Box<T>,
}

impl<T> CoerceUnsized<Foo<dyn Any>> for Foo<T> {}
//~^ ERROR the parameter type `T` may not live long enough

fn main() {}
