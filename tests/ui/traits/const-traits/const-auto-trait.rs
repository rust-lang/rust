//@ compile-flags: -Znext-solver
// See rust-lang/rust#149285 for this test

#![feature(auto_traits, const_trait_impl)]

const auto trait Marker {}
//~^ ERROR: auto traits cannot be const

fn scope() {
    fn check<T: const Marker>() {}
    check::<()>();
    //~^ ERROR: the trait bound `(): const Marker` is not satisfied
}

fn main() {}
