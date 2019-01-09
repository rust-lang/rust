// run-pass

// ignore-wasm32-bare compiled with panic=abort by default

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::panic;

fn main() {
    let mut foo = || {
        if true {
            return
        }
        yield;
    };

    match unsafe { foo.resume() } {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }

    match panic::catch_unwind(move || unsafe { foo.resume() }) {
        Ok(_) => panic!("generator successfully resumed"),
        Err(_) => {}
    }
}
