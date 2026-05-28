//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo;

impl Foo {
    type Bar<T> = u8;
}

fn main() {
    let a: Foo::Bar<()>;
}
