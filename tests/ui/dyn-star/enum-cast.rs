//@ check-pass

// This used to ICE, because the compiler confused a pointer-like to dyn* coercion
// with a c-like enum to integer cast.

#![feature(dyn_star)]
#![expect(incomplete_features)]

enum E {
    Num(usize),
}

trait Trait {}
impl Trait for E {}

fn main() {
    let _ = E::Num(42) as dyn* Trait;
}
