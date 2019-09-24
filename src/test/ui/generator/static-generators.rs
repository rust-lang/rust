// run-pass

#![feature(generators, generator_trait)]

use std::pin::Pin;
use std::ops::{Generator, GeneratorState};

fn main() {
    let mut generator = static || {
        let a = true;
        let b = &a;
        yield;
        assert_eq!(b as *const _, &a as *const _);
    };
    // Safety: We shadow the original generator variable so have no safe API to
    // move it after this point.
    let mut generator = unsafe { Pin::new_unchecked(&mut generator) };
    assert_eq!(generator.as_mut().resume(), GeneratorState::Yielded(()));
    assert_eq!(generator.as_mut().resume(), GeneratorState::Complete(()));
}
