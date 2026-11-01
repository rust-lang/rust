//@ compile-flags: --emit metadata
#![feature(coroutines, coroutine_trait)]

use std::marker::Unpin;
use std::ops::Coroutine;

pub fn g() -> impl Coroutine<(), Yield = (), Return = ()> {
    #[coroutine]
    || {
        yield;
    }
}
