//@ check-pass
#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo;

impl Foo {
    type Bar = isize;
}

fn main() {
    let x: Foo::Bar;
    x = 0isize;
}
