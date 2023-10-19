#![feature(impl_trait_in_assoc_type, generator_trait, generators)]
use std::ops::Coroutine;

trait Runnable {
    type Gen: Coroutine<Yield = (), Return = ()>;

    fn run(&mut self) -> Self::Gen;
}

struct Implementor {}

impl Runnable for Implementor {
    type Gen = impl Coroutine<Yield = (), Return = ()>;

    fn run(&mut self) -> Self::Gen {
        //~^ ERROR: type mismatch resolving
        move || {
            yield 1;
        }
    }
}

fn main() {}
