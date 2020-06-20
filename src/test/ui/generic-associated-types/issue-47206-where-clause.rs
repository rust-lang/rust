// Check that this program doesn't cause the compiler to error without output.

#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait Foo {
    type Assoc3<T>;
}

struct Bar;

impl Foo for Bar {
    type Assoc3<T> where T: Iterator = Vec<T>;
    //~^ ERROR impl has stricter requirements than trait
}

fn main() {}
