//@ run-pass

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let _coroutine = #[coroutine]
    || {
        let mut sub_coroutine = #[coroutine]
        || {
            2.yield;
        };

        match Pin::new(&mut sub_coroutine).resume(()) {
            CoroutineState::Yielded(x) => {
                x.yield;
            }
            _ => panic!(),
        };
    };
}
