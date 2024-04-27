//@ run-pass

//@ aux-build:xcrate.rs

#![feature(coroutines, coroutine_trait)]

extern crate xcrate;

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut foo = xcrate::foo();

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }

    let mut foo = xcrate::bar(3);

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Yielded(3) => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}
