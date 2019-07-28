// run-pass
#![allow(dead_code)]

#![feature(specialization)]

// Make sure we *can* project non-defaulted associated types
// cf compile-fail/specialization-default-projection.rs

// First, do so without any use of specialization

trait Foo {
    type Assoc;
}

impl<T> Foo for T {
    type Assoc = ();
}

fn generic_foo<T>() -> <T as Foo>::Assoc {
    ()
}

// Next, allow for one layer of specialization

trait Bar {
    type Assoc;
}

default impl<T> Bar for T {
    type Assoc = ();
}

impl<T: Clone> Bar for T {
    type Assoc = u8;
}

fn generic_bar_clone<T: Clone>() -> <T as Bar>::Assoc {
    0u8
}

fn main() {
}
