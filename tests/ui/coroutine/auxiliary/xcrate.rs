#![feature(coroutines, coroutine_trait)]

use std::marker::Unpin;
use std::ops::Coroutine;

pub fn foo() -> impl Coroutine<(), Yield = (), Return = ()> {
    || {
        if false {
            yield;
        }
    }
}

pub fn bar<T: 'static>(t: T) -> Box<Coroutine<(), Yield = T, Return = ()> + Unpin> {
    Box::new(|| {
        yield t;
    })
}
