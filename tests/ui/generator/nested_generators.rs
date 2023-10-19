// run-pass

#![feature(generators, generator_trait)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let _generator = || {
        let mut sub_generator = || {
            yield 2;
        };

        match Pin::new(&mut sub_generator).resume(()) {
            CoroutineState::Yielded(x) => {
                yield x;
            }
            _ => panic!(),
        };
    };
}
