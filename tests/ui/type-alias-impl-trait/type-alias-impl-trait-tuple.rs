//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

pub trait MyTrait {}

impl MyTrait for bool {}

pub type Foo = impl MyTrait;

#[define_opaque(Foo)]
pub fn make_foo() -> Foo {
    true
}

struct Blah {
    my_foo: Foo,
    my_u8: u8,
}

impl Blah {
    fn new() -> Blah {
        Blah { my_foo: make_foo(), my_u8: 12 }
    }
    fn into_inner(self) -> (Foo, u8, Foo) {
        (self.my_foo, self.my_u8, make_foo())
    }
}

fn main() {}
