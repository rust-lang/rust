//@ check-pass

#![feature(const_convert)]
#![feature(const_trait_impl)]

use std::ops::Deref;

struct Wrap<T>(T);
struct Foo;

impl Foo {
    const fn call(&self) {}
}

impl<T> const Deref for Wrap<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const fn foo() {
    let x = Wrap(Foo);
    x.call();
}

fn main() {}
