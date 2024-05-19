#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    run_coroutine::<i32>();
}

fn run_coroutine<T>() {
    let mut coroutine = #[coroutine]
    || {
        yield;
        return;
    };
    Pin::new(&mut coroutine).resume(());
}
