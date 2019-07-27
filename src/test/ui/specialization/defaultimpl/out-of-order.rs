// run-pass

// Test that you can list the more specific impl before the more general one.

#![feature(specialization)]

trait Foo {
    type Out;
}

impl Foo for bool {
    type Out = ();
}

default impl<T> Foo for T {
    type Out = bool;
}

fn main() {}
