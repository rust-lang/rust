//@ run-pass

use std::marker;

pub struct Foo<T>(marker::PhantomData<T>);

impl<T> Iterator for Foo<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        None
    }
}

impl<T> Drop for Foo<T> {
    fn drop(&mut self) {
        self.next();
    }
}

pub fn foo<'a>(_: Foo<&'a ()>) {}

pub fn main() {}
