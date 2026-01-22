//@ check-pass
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

// Regression test for normalizing const projections
// with associated const equality bounds.

trait Trait {
    #[type_const]
    const C: usize;
}

fn f<T: Trait<C = 1>>() {
    // This must normalize <T as Trait>::C to 1
    let _: [(); T::C] = [()];
}

fn main() {}
