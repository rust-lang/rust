// run-pass

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};

struct W<T>(T);

// This impl isn't safe in general, but the generator used in this test is movable
// so it won't cause problems.
impl<T: Generator<Return = ()>> Iterator for W<T> {
    type Item = T::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        match unsafe { self.0.resume() } {
            GeneratorState::Complete(..) => None,
            GeneratorState::Yielded(v) => Some(v),
        }
    }
}

fn test() -> impl Generator<Return=(), Yield=u8> {
    || {
        for i in 1..6 {
            yield i
        }
    }
}

fn main() {
    let end = 11;

    let closure_test = |start| {
        move || {
            for i in start..end {
                yield i
            }
        }
    };

    assert!(W(test()).chain(W(closure_test(6))).eq(1..11));
}
