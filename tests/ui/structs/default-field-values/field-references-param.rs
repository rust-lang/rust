//@ build-pass

#![feature(default_field_values)]

struct W<const X: usize>;

impl<const X: usize> W<X> {
    const fn new() -> Self { W }
}

struct Z<const X: usize> {
    // No inference.
    one: W<X> = W::<X>::new(),

    // Inference works too.
    two: W<X> = W::new(),

    // An anon const that is too generic before substitution.
    too_generic: usize = X + 1,
}

fn use_generically<const X: usize>() {
    let x: Z<X> = Z { .. };
}

fn main() {
    let x: Z<0> = Z { .. };
    use_generically::<0>();
}
