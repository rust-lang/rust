// Regression test for #93182: trait const default CONST used to ICE in ArgFolder.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub const CONST: usize = 64;
pub trait Tr<const S: usize = CONST>: Foo<A<S>> {}

struct St();

struct A<const S: usize>([u8; S]);

pub trait Foo<T> {
    fn foo(_: T);
}

impl<const S: usize> Foo<A<S>> for St {
    fn foo(_: A<S>) {
        todo!()
    }
}

pub trait FooBar {
    type Tr: Tr;
    //~^ ERROR mismatched types
}

pub fn main() {}
