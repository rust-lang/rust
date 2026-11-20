#![feature(coroutines, coroutine_clone, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;

pub fn make() -> impl Coroutine<Yield = i32, Return = ()> + Clone + Unpin {
    #[coroutine]
    || {
        yield 1;
    }
}
