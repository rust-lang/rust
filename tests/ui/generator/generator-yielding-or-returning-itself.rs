#![feature(generator_trait)]
#![feature(generators)]

// Test that we cannot create a generator that returns a value of its
// own type.

use std::ops::Coroutine;

pub fn want_cyclic_generator_return<T>(_: T)
    where T: Coroutine<Yield = (), Return = T>
{
}

fn supply_cyclic_generator_return() {
    want_cyclic_generator_return(|| {
        //~^ ERROR type mismatch
        if false { yield None.unwrap(); }
        None.unwrap()
    })
}

pub fn want_cyclic_generator_yield<T>(_: T)
    where T: Coroutine<Yield = T, Return = ()>
{
}

fn supply_cyclic_generator_yield() {
    want_cyclic_generator_yield(|| {
        //~^ ERROR type mismatch
        if false { yield None.unwrap(); }
        None.unwrap()
    })
}

fn main() { }
