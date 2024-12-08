//@ check-pass
#![allow(dead_code)]
use std::ops::{Deref, DerefMut};

struct Foo;

impl Foo {
    fn foo_mut(&mut self) {}
}

struct Bar(Foo);

impl Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &self.0
    }
}

impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Foo {
        &mut self.0
    }
}

fn test(mut bar: Box<Bar>) {
    bar.foo_mut();
}

fn main() {}
