// Check that repeating a diverging constant 0 times doesn't lead to unsoundness.
//
// This specific test checks that using an associated const that evaluates to 0, rather than just
// plain literal 0, doesn't allow the unsoundness either.
//
// Issue: <https://github.com/rust-lang/rust/issues/165323>
//
//@ build-fail
#![feature(gca_min_const_items)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const ZERO: usize;
}

impl Trait for () {
    const ZERO: usize = gca!(0);
}

fn ex_nihilo<T: Trait>() -> u32 {
    [const { panic!() } as !; gca!(T::ZERO)];
    //~^ error: evaluation panicked: explicit panic
}

fn main() {
    dbg!(ex_nihilo::<()>());
}
