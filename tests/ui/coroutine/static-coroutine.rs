//@ run-pass

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut coroutine = #[coroutine]
    static || {
        let a = true;
        let b = &a;
        yield;
        assert_eq!(b as *const _, &a as *const _);
    };
    // SAFETY: We shadow the original coroutine variable so have no safe API to
    // move it after this point.
    let mut coroutine = unsafe { Pin::new_unchecked(&mut coroutine) };
    assert_eq!(coroutine.as_mut().resume(()), CoroutineState::Yielded(()));
    assert_eq!(coroutine.as_mut().resume(()), CoroutineState::Complete(()));
}
