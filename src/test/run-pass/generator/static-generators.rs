// run-pass

#![feature(generators, generator_trait)]

use std::ops::{Generator, GeneratorState};

fn main() {
    let mut generator = static || {
        let a = true;
        let b = &a;
        yield;
        assert_eq!(b as *const _, &a as *const _);
    };
    unsafe {
        assert_eq!(generator.resume(), GeneratorState::Yielded(()));
        assert_eq!(generator.resume(), GeneratorState::Complete(()));
    }
}
