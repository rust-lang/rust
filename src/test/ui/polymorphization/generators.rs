// build-fail
// compile-flags:-Zpolymorphize=on
#![feature(const_generics, generators, generator_trait, rustc_attrs)]
//~^ WARN the feature `const_generics` is incomplete

use std::marker::Unpin;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

enum YieldOrReturn<Y, R> {
    Yield(Y),
    Return(R),
}

fn finish<T, Y, R>(mut t: T) -> Vec<YieldOrReturn<Y, R>>
where
    T: Generator<(), Yield = Y, Return = R> + Unpin,
{
    let mut results = Vec::new();
    loop {
        match Pin::new(&mut t).resume(()) {
            GeneratorState::Yielded(yielded) => results.push(YieldOrReturn::Yield(yielded)),
            GeneratorState::Complete(returned) => {
                results.push(YieldOrReturn::Return(returned));
                return results;
            }
        }
    }
}

// This test checks that the polymorphization analysis functions on generators.

#[rustc_polymorphize_error]
pub fn unused_type<T>() -> impl Generator<(), Yield = u32, Return = u32> + Unpin {
    //~^ ERROR item has unused generic parameters
    || {
        //~^ ERROR item has unused generic parameters
        yield 1;
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_type_in_yield<Y: Default>() -> impl Generator<(), Yield = Y, Return = u32> + Unpin {
    || {
        yield Y::default();
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_type_in_return<R: Default>() -> impl Generator<(), Yield = u32, Return = R> + Unpin {
    || {
        yield 3;
        R::default()
    }
}

#[rustc_polymorphize_error]
pub fn unused_const<const T: u32>() -> impl Generator<(), Yield = u32, Return = u32> + Unpin {
    //~^ ERROR item has unused generic parameters
    || {
        //~^ ERROR item has unused generic parameters
        yield 1;
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_const_in_yield<const Y: u32>() -> impl Generator<(), Yield = u32, Return = u32> + Unpin
{
    || {
        yield Y;
        2
    }
}

#[rustc_polymorphize_error]
pub fn used_const_in_return<const R: u32>() -> impl Generator<(), Yield = u32, Return = u32> + Unpin
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
