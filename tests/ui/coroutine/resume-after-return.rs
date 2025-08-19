//@ ignore-backends: gcc
//@ run-pass
//@ needs-unwind

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::panic;
use std::pin::Pin;

fn main() {
    let mut foo = #[coroutine]
    || {
        if true {
            return;
        }
        yield;
    };

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }

    match panic::catch_unwind(move || Pin::new(&mut foo).resume(())) {
        Ok(_) => panic!("coroutine successfully resumed"),
        Err(_) => {}
    }
}
