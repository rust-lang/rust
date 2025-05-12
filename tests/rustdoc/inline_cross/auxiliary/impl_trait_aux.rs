//@ edition:2018

use std::ops::Deref;

pub fn func<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}

pub fn func2<T>(
    _x: impl Deref<Target = Option<T>> + Iterator<Item = T>,
    _y: impl Iterator<Item = u8>,
) {}

pub fn func3(_x: impl Iterator<Item = impl Iterator<Item = u8>> + Clone) {}

pub fn func4<T: Iterator<Item = impl Clone>>(_x: T) {}

pub fn func5(
    _f: impl for<'any> Fn(&'any str, &'any str) -> bool + for<'r> Other<T<'r> = ()>,
    _a: impl for<'beta, 'alpha, '_gamma> Auxiliary<'alpha, Item<'beta> = fn(&'beta ())>,
) {}

pub trait Other {
    type T<'dependency>;
}

pub trait Auxiliary<'arena> {
    type Item<'input>;
}

pub async fn async_fn() {}

pub struct Foo;

impl Foo {
    pub fn method<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}
}
