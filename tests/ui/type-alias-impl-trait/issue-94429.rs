#![feature(impl_trait_in_assoc_type, coroutine_trait, coroutines)]
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
