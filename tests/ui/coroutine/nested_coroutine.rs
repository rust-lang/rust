//@ run-pass

#![feature(coroutines, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let _coroutine = || {
        let mut sub_coroutine = || {
            yield 2;
        };

        match Pin::new(&mut sub_coroutine).resume(()) {
            CoroutineState::Yielded(x) => {
                yield x;
            }
            _ => panic!(),
        };
    };
}
