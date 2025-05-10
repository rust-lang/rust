//@ run-pass

#![feature(coroutines)]
#![feature(coroutine_trait)]
#![feature(stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::pin;

fn coro() {
    let c = #[coroutine]
    static || {
        let mut a = 19;
        let b = &mut a;
        yield;
        *b = 23;
        yield;
        a
    };

    let mut c = pin!(c);
    assert_eq!(c.as_mut().resume(()), CoroutineState::Yielded(()));
    assert_eq!(c.as_mut().resume(()), CoroutineState::Yielded(()));
    assert_eq!(c.as_mut().resume(()), CoroutineState::Complete(23));
}

fn main() {
    coro();
}
