//! Tests that we cannot produce a generator that accepts a resume argument
//! with any lifetime and then stores it across a `yield`.

#![feature(generators, generator_trait)]

use std::ops::Generator;

fn test(a: impl for<'a> Generator<&'a mut bool>) {}

fn main() {
    let gen = |arg: &mut bool| {
        yield ();
        *arg = true;
    };
    test(gen);
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}
