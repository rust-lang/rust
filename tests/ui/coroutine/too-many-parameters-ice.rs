//! Regression test for <https://github.com/rust-lang/rust/issues/139570>.
//! A coroutine with too many parameters should emit errors without an ICE.

#![feature(coroutines)]

fn main() {
    #[coroutine]
    |(1, 42), ()| {
        //~^ ERROR too many parameters for a coroutine
        //~| ERROR refutable pattern in closure argument
        yield
    };
}
