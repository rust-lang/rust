// issue 65419 - Attempting to run an `async fn` after completion mentions coroutines when it should
// be talking about `async fn`s instead. Regression test added to make sure coroutines still
// panic when resumed after completion.

//@ run-fail
//@ error-pattern:coroutine resumed after completion
//@ edition:2018
//@ ignore-wasm no panic or subprocess support
//@ ignore-emscripten no panic or subprocess support

#![feature(coroutines, coroutine_trait)]

use std::{
    ops::Coroutine,
    pin::Pin,
};

fn main() {
    let mut g = || {
        yield;
    };
    Pin::new(&mut g).resume(()); // Yields once.
    Pin::new(&mut g).resume(()); // Completes here.
    Pin::new(&mut g).resume(()); // Panics here.
}
