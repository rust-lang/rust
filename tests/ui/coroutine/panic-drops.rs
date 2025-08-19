//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::panic;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

static A: AtomicUsize = AtomicUsize::new(0);

struct B;

impl Drop for B {
    fn drop(&mut self) {
        A.fetch_add(1, Ordering::SeqCst);
    }
}

fn bool_true() -> bool {
    true
}

fn main() {
    let b = B;
    let mut foo = #[coroutine]
    || {
        if bool_true() {
            panic!();
        }
        drop(b);
        yield;
    };

    assert_eq!(A.load(Ordering::SeqCst), 0);
    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| Pin::new(&mut foo).resume(())));
    assert!(res.is_err());
    assert_eq!(A.load(Ordering::SeqCst), 1);

    let mut foo = #[coroutine]
    || {
        if bool_true() {
            panic!();
        }
        drop(B);
        yield;
    };

    assert_eq!(A.load(Ordering::SeqCst), 1);
    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| Pin::new(&mut foo).resume(())));
    assert!(res.is_err());
    assert_eq!(A.load(Ordering::SeqCst), 1);
}
