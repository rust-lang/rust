//@ check-pass

#![feature(default_field_values)]

struct W<const X: usize>;

impl<const X: usize> W<X> {
    const fn new() -> Self { W }
}

struct Z<const X: usize> {
    one: W<X> = W::<X>::new(),
    // Inference works too.
    two: W<X> = W::new(),
}

fn main() {}
