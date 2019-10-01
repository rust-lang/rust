// edition:2018

use std::ops::Deref;

pub fn func<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}

pub fn func2<T>(
    _x: impl Deref<Target = Option<T>> + Iterator<Item = T>,
    _y: impl Iterator<Item = u8>,
) {}

pub fn func3(_x: impl Iterator<Item = impl Iterator<Item = u8>> + Clone) {}

pub fn func4<T: Iterator<Item = impl Clone>>(_x: T) {}

pub async fn async_fn() {}

pub struct Foo;

impl Foo {
    pub fn method<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}
}

pub struct Bar;

impl Bar {
    pub async fn async_foo(&self) {}
}
