//@ known-bug: #93182
#![feature(generic_const_exprs)]

// this causes an ICE!!!
pub const CONST: usize = 64;
pub trait Tr<const S: usize = CONST>: Foo<A<S>> {}

// no ICE
// pub trait Digest<const S: usize = 64>: FromH<[u8; S]> {}

struct St ();

struct A<const S: usize> ([u8; S]);

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
}

pub fn main() {}
