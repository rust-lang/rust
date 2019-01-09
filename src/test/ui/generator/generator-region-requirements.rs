// revisions: ast nll
// ignore-compare-mode-nll

#![feature(generators, generator_trait)]
#![cfg_attr(nll, feature(nll))]
use std::ops::{Generator, GeneratorState};

fn dangle(x: &mut i32) -> &'static mut i32 {
    let mut g = || {
        yield;
        x
    };
    loop {
        match unsafe { g.resume() } {
            GeneratorState::Complete(c) => return c,
//[nll]~^ ERROR explicit lifetime required
//[ast]~^^ ERROR explicit lifetime required
            GeneratorState::Yielded(_) => (),
        }
    }
}

fn main() {}
