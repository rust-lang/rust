#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#![deny(unused)]

struct Foo;

impl Foo {
    fn one() {}
    //~^ ERROR associated items `one`, `two`, `CONSTANT`, `Type`, and `three` are never used [dead_code]

    fn two(&self) {}

    // seperation between items
    // ...
    // ...

    fn used() {}

    const CONSTANT: usize = 5;

    // more seperation
    // ...
    // ...

    type Type = usize;

    fn three(&self) {
        Foo::one();
        // ...
    }
}

fn main() {
    Foo::used();
}
