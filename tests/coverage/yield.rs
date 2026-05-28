#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
#![allow(unused_assignments)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut coroutine = #[coroutine]
    || {
        yield 1;
        return "foo";
    };

    match Pin::new(&mut coroutine).resume(()) {
        CoroutineState::Yielded(1) => {}
        _ => panic!("unexpected value from resume"),
    }
    match Pin::new(&mut coroutine).resume(()) {
        CoroutineState::Complete("foo") => {}
        _ => panic!("unexpected value from resume"),
    }

    let mut coroutine = #[coroutine]
    || {
        yield 1;
        yield 2;
        yield 3;
        return "foo";
    };

    match Pin::new(&mut coroutine).resume(()) {
        CoroutineState::Yielded(1) => {}
        _ => panic!("unexpected value from resume"),
    }
    match Pin::new(&mut coroutine).resume(()) {
        CoroutineState::Yielded(2) => {}
        _ => panic!("unexpected value from resume"),
    }
}
