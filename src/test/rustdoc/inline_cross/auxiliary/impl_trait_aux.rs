use std::ops::Deref;

pub fn func<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}

pub fn func2<T>(_x: impl Deref<Target = Option<T>> + Iterator<Item = T>, _y: impl Iterator<Item = u8>) {}

pub struct Foo;

impl Foo {
    pub fn method<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}
}
