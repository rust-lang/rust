//@ run-pass

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let _coroutine = #[coroutine]
    || {
        let mut sub_coroutine = #[coroutine]
        || {
            yield 2;
        };

        match Pin::new(&mut sub_coroutine).resume(()) {
            CoroutineState::Yielded(x) => {
                yield x;
            }
            _ => panic!(),
        };
    };
}
