#![feature(coroutine_trait)]
#![feature(coroutines, stmt_expr_attributes)]

// Test that we cannot create a coroutine that returns a value of its
// own type.

use std::ops::Coroutine;

pub fn want_cyclic_coroutine_return<T>(_: T)
    where T: Coroutine<Yield = (), Return = T>
{
}

fn supply_cyclic_coroutine_return() {
    want_cyclic_coroutine_return(#[coroutine] || {
        //~^ ERROR type mismatch
        if false { yield None.unwrap(); }
        None.unwrap()
    })
}

pub fn want_cyclic_coroutine_yield<T>(_: T)
    where T: Coroutine<Yield = T, Return = ()>
{
}

fn supply_cyclic_coroutine_yield() {
    want_cyclic_coroutine_yield(#[coroutine] || {
        //~^ ERROR type mismatch
        if false { yield None.unwrap(); }
        None.unwrap()
    })
}

fn main() { }
