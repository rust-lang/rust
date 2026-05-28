// Regression test for issue #109759.
//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo;

struct Bar<const X: usize>([(); X]);

impl<const X: usize> Bar<X> {
    pub fn new() -> Self {
        Self([(); X])
    }
}

impl Foo {
    type Bar<const X: usize> = Bar<X>;
}

fn main() {
    let _ = Foo::Bar::<10>::new();
}
