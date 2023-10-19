#![feature(generators, generator_trait)]
#![allow(unused_assignments)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut generator = || {
        yield 1;
        return "foo";
    };

    match Pin::new(&mut generator).resume(()) {
        CoroutineState::Yielded(1) => {}
        _ => panic!("unexpected value from resume"),
    }
    match Pin::new(&mut generator).resume(()) {
        CoroutineState::Complete("foo") => {}
        _ => panic!("unexpected value from resume"),
    }

    let mut generator = || {
        yield 1;
        yield 2;
        yield 3;
        return "foo";
    };

    match Pin::new(&mut generator).resume(()) {
        CoroutineState::Yielded(1) => {}
        _ => panic!("unexpected value from resume"),
    }
    match Pin::new(&mut generator).resume(()) {
        CoroutineState::Yielded(2) => {}
        _ => panic!("unexpected value from resume"),
    }
}
