#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

fn msg() -> u32 {
    0
}

pub fn foo() -> impl Coroutine<(), Yield = (), Return = u32> {
    #[coroutine]
    || {
        yield;
        return msg();
    }
}
