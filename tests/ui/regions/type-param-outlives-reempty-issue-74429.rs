// Regression test for #74429, where we didn't think that a type parameter
// outlived `ReEmpty`.

//@ check-pass

#![allow(dropping_copy_types)]

use std::marker::PhantomData;

fn apply<T, F: FnOnce(T)>(_: T, _: F) {}

#[derive(Clone, Copy)]
struct Invariant<T> {
    t: T,
    p: PhantomData<fn(T) -> T>,
}

fn verify_reempty<T>(x: T) {
    // r is inferred to have type `Invariant<&ReEmpty(U0) T>`
    let r = Invariant { t: &x, p: PhantomData };
    // Creates a new universe, all variables from now on are in `U1`, say.
    let _: fn(&()) = |_| {};
    // Closure parameter is of type `&ReEmpty(U1) T`, so the closure has an implied
    // bound of `T: ReEmpty(U1)`
    apply(&x, |_| {
        // Requires `typeof(r)` is well-formed, i.e. `T: ReEmpty(U0)`. If we
        // only have the implied bound from the closure parameter to use this
        // requires `ReEmpty(U1): ReEmpty(U0)`, which isn't true so we reported
        // an error.
        //
        // This doesn't happen any more because we ensure that `T: ReEmpty(U0)`
        // is an implicit bound for all type parameters.
        drop(r);
    });
}

fn main() {}
