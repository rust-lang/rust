#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    run_coroutine::<i32>();
}

fn run_coroutine<T>() {
    let mut coroutine = || {
        yield;
        return;
    };
    Pin::new(&mut coroutine).resume(());
}
