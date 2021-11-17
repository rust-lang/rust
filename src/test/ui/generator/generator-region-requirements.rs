#![feature(generators, generator_trait)]
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

fn dangle(x: &mut i32) -> &'static mut i32 {
    let mut g = || {
        yield;
        x
        //~^ ERROR `x` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
    };
    loop {
        match Pin::new(&mut g).resume(()) {
            GeneratorState::Complete(c) => return c,
            GeneratorState::Yielded(_) => (),
        }
    }
}

fn main() {}
