//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[current] check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

pub trait MyTrait {}

impl MyTrait for bool {}

type Foo = impl MyTrait;

struct Blah {
    my_foo: Foo,
    my_u8: u8,
}

impl Blah {
    fn new() -> Blah {
        Blah { my_foo: make_foo(), my_u8: 12 }
        //[next]~^ ERROR type annotations needed: cannot satisfy `Foo == _`
    }
    fn into_inner(self) -> (Foo, u8, Foo) {
        (self.my_foo, self.my_u8, make_foo())
        //[next]~^ ERROR type annotations needed: cannot satisfy `Foo == _`
    }
}

fn make_foo() -> Foo {
    true
}

fn main() {}
