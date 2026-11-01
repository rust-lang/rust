#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo;

impl Foo {
    type Baz; //~ ERROR associated type in `impl` without body
}

fn main() {}
