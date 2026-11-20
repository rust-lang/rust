//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;
use std::panic;

fn main() {
    let mut foo = #[coroutine] || {
        if true {
            panic!();
        }
        yield;
    };

    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        Pin::new(&mut foo).resume(())
    }));
    assert!(res.is_err());

    for _ in 0..10 {
        let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            Pin::new(&mut foo).resume(())
        }));
        assert!(res.is_err());
    }
}
