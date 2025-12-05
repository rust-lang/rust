#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn dangle(x: &mut i32) -> &'static mut i32 {
    let mut g = #[coroutine] || {
        yield;
        x
    };
    loop {
        match Pin::new(&mut g).resume(()) {
            CoroutineState::Complete(c) => return c,
            //~^ ERROR lifetime may not live long enough
            CoroutineState::Yielded(_) => (),
        }
    }
}

fn main() {}
