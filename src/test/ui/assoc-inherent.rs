// Test that inherent associated types work with
// inherent_associated_types feature gate.

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo;

impl Foo {
    type Bar = isize;
}

impl Foo {
    type Baz; //~ ERROR associated type in `impl` without body
}

fn main() {
    let x : Foo::Bar; //~ERROR ambiguous associated type
    x = 0isize;
}
