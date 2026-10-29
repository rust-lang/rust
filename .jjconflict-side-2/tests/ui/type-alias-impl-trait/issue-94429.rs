#![feature(impl_trait_in_assoc_type, coroutine_trait, coroutines)]
use std::ops::Coroutine;

trait Runnable {
    type Coro: Coroutine<Yield = (), Return = ()>;

    fn run(&mut self) -> Self::Coro;
}

struct Implementor {}

impl Runnable for Implementor {
    type Coro = impl Coroutine<Yield = (), Return = ()>;

    fn run(&mut self) -> Self::Coro {
        //~^ ERROR: type mismatch resolving
        #[coroutine]
        move || {
            yield 1;
        }
    }
}

fn main() {}
