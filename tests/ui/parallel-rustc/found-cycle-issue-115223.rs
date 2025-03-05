// Test for #115223, which causes a deadlock bug without finding the cycle
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=16
//@ run-pass

#![crate_name = "foo"]

use std::ops;

pub struct Foo;

impl Foo {
    pub fn foo(&mut self) {}
}

pub struct Bar {
    foo: Foo,
}

impl ops::Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &self.foo
    }
}

impl ops::DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Foo {
        &mut self.foo
    }
}

fn main() {}
