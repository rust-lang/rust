//@ compile-flags: -Znext-solver
#![feature(const_convert, never_type, const_trait_impl)]

const fn impls_from<T: [const] From<!>>() {}

const fn foo() {
    // This previously ICE'd when encountering the reservation impl
    // from the standard library.
    impls_from::<()>();
    //~^ ERROR the trait bound `(): From<!>` is not satisfied
}

fn main() {}
