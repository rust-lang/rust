// Testing inference capabilities.
// check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

use std::convert::identity;

struct Container<T>(T);

impl Container<u32> {
    type Sink = ();
}

impl<Any> Container<Any> {
    type Thing = Any;
}

impl<T> Container<(T, ())> {
    type Output = ((), Wrapped<T>);
}

fn main() {
    // Inferred via the Self type of the impl.
    let _: Container<_>::Sink;

    // Inferred via the RHS:

    let _: Container<_>::Thing = 0;

    let _: Container<Wrapped<_>>::Thing = Wrapped(false);

    let _: Container<_>::Output = (drop(1), Wrapped("..."));

    let binding: Container<_>::Thing = Default::default(); // unsolved at this point
    identity::<String>(binding); // constrained and solved here
}

struct Wrapped<T>(T);
