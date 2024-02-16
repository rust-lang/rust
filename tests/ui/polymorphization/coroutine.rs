//@ build-fail
//@ compile-flags:-Zpolymorphize=on -Zinline-mir=off
#![feature(generic_const_exprs, coroutines, coroutine_trait, rustc_attrs)]
//~^ WARN the feature `generic_const_exprs` is incomplete

use std::marker::Unpin;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

enum YieldOrReturn<Y, R> {
    Yield(Y),
    Return(R),
}

fn finish<T, Y, R>(mut t: T) -> Vec<YieldOrReturn<Y, R>>
where
    T: Coroutine<(), Yield = Y, Return = R> + Unpin,
{
    let mut results = Vec::new();
    loop {
        match Pin::new(&mut t).resume(()) {
            CoroutineState::Yielded(yielded) => results.push(YieldOrReturn::Yield(yielded)),
            CoroutineState::Complete(returned) => {
                results.push(YieldOrReturn::Return(returned));
                return results;
            }
        }
    }
}

// This test checks that the polymorphization analysis functions on coroutines.

#[rustc_polymorphize_error]
pub fn unused_type<T>() -> impl Coroutine<(), Yield = u32, Return = u32> + Unpin {
    || {
        //~^ ERROR item has unused generic parameters
        yield 1;
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_type_in_yield<Y: Default>() -> impl Coroutine<(), Yield = Y, Return = u32> + Unpin {
    || {
        yield Y::default();
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_type_in_return<R: Default>() -> impl Coroutine<(), Yield = u32, Return = R> + Unpin {
    || {
        yield 3;
        R::default()
    }
}

#[rustc_polymorphize_error]
pub fn unused_const<const T: u32>() -> impl Coroutine<(), Yield = u32, Return = u32> + Unpin {
    || {
        //~^ ERROR item has unused generic parameters
        yield 1;
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_const_in_yield<const Y: u32>() -> impl Coroutine<(), Yield = u32, Return = u32> + Unpin
{
    || {
        yield Y;
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_const_in_return<const R: u32>() -> impl Coroutine<(), Yield = u32, Return = u32> + Unpin
{
    || {
        yield 4;
        R
    }
}

fn main() {
    finish(unused_type::<u32>());
    finish(used_type_in_yield::<u32>());
    finish(used_type_in_return::<u32>());
    finish(unused_const::<1u32>());
    finish(used_const_in_yield::<1u32>());
    finish(used_const_in_return::<1u32>());
}
