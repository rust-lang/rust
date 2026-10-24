#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

fn msg() -> u32 {
    0
}

pub fn foo() -> impl for<'y> Coroutine<(), Yield<'y> = (), Return = u32> {
    #[coroutine]
    || {
        yield;
        return msg();
    }
}
