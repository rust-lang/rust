// check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

pub trait MyTrait {}

impl MyTrait for bool {}

struct Blah {
    my_foo: Foo,
    my_u8: u8
}

impl Blah {
    fn new() -> Blah {
        Blah {
            my_foo: make_foo(),
            my_u8: 12
        }
    }
    fn into_inner(self) -> (Foo, u8) {
        (self.my_foo, self.my_u8)
    }
}

fn make_foo() -> Foo {
    true
}

type Foo = impl MyTrait;

fn main() {}
