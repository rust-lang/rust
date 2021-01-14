// compile-flags: --emit metadata
#![feature(generators, generator_trait)]

use std::marker::Unpin;
use std::ops::Generator;

pub fn g() -> impl Generator<(), Yield = (), Return = ()> {
    || {
        yield;
    }
}
