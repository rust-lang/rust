// compile-flags: --emit metadata
#![feature(generators, generator_trait)]

use std::marker::Unpin;
use std::ops::Coroutine;

pub fn g() -> impl Coroutine<(), Yield = (), Return = ()> {
    || {
        yield;
    }
}
