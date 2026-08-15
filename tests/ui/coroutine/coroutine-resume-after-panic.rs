//@ run-fail
//@ needs-unwind
//@ error-pattern:coroutine resumed after panicking
//@ needs-subprocess

// Test that we get the correct message for resuming a panicked coroutine.

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::{
    ops::Coroutine,
    pin::Pin,
    panic,
};

fn main() {
    let mut g = #[coroutine] || {
        panic!();
        yield;
    };
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let x = Pin::new(&mut g).resume(());
    }));
    Pin::new(&mut g).resume(());
}
