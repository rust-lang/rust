#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::marker::Unpin;
use std::ops::Coroutine;

pub fn foo() -> impl for<'y> Coroutine<(), Yield<'y> = (), Return = ()> {
    #[coroutine]
    || {
        if false {
            yield;
        }
    }
}

pub fn bar<T: 'static>(t: T) -> Box<dyn Coroutine<(), Yield = T, Return = ()> + Unpin> {
    Box::new(
        #[coroutine]
        || {
            yield t;
        },
    )
}
