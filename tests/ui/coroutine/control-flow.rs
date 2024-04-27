//@ run-pass

//@ revisions: default nomiropt
//@[nomiropt]compile-flags: -Z mir-opt-level=0

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{CoroutineState, Coroutine};
use std::pin::Pin;

fn finish<T>(mut amt: usize, mut t: T) -> T::Return
    where T: Coroutine<(), Yield = ()> + Unpin,
{
    loop {
        match Pin::new(&mut t).resume(()) {
            CoroutineState::Yielded(()) => amt = amt.checked_sub(1).unwrap(),
            CoroutineState::Complete(ret) => {
                assert_eq!(amt, 0);
                return ret
            }
        }
    }

}

fn main() {
    finish(1, #[coroutine] || yield);
    finish(8, #[coroutine] || {
        for _ in 0..8 {
            yield;
        }
    });
    finish(1, #[coroutine] || {
        if true {
            yield;
        } else {
        }
    });
    finish(1, #[coroutine] || {
        if false {
        } else {
            yield;
        }
    });
    finish(2, #[coroutine] || {
        if { yield; false } {
            yield;
            panic!()
        }
        yield
    });
}
