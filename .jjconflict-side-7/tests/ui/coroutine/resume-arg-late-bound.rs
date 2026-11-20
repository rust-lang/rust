//! Tests that we cannot produce a coroutine that accepts a resume argument
//! with any lifetime and then stores it across a `yield`.

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;

fn test(a: impl for<'a> Coroutine<&'a mut bool>) {}

fn main() {
    let gen = #[coroutine] |arg: &mut bool| {
        yield ();
        *arg = true;
    };
    test(gen);
    //~^ ERROR implementation of `Coroutine` is not general enough
}
