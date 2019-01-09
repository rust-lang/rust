// run-pass

// aux-build:xcrate.rs

#![feature(generators, generator_trait)]

extern crate xcrate;

use std::ops::{GeneratorState, Generator};

fn main() {
    let mut foo = xcrate::foo();

    match unsafe { foo.resume() } {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }

    let mut foo = xcrate::bar(3);

    match unsafe { foo.resume() } {
        GeneratorState::Yielded(3) => {}
        s => panic!("bad state: {:?}", s),
    }
    match unsafe { foo.resume() } {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}
