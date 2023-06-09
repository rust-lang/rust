//! Test that we get the expected type mismatch error instead of "closure is expected to take 0
//! arguments" (which got introduced after implementing resume arguments).

#![feature(generators, generator_trait)]

use std::ops::Generator;

fn f<G: Generator>(_: G, _: G::Return) {}

fn main() {
    f(
        |a: u8| {
            if false {
                yield ();
            } else {
                a
                //~^ error: `if` and `else` have incompatible types
            }
        },
        0u8,
    );
}
