// This demonstrates a proposed alternate or additional option of having yield in postfix position.

//@ run-pass
//@ edition: 2024

#![feature(gen_blocks, coroutines, coroutine_trait, yield_expr, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::pin;

fn main() {
    // generators (i.e. yield doesn't return anything useful)
    let mut gn = gen {
        yield 1;
        2.yield;
    };

    assert_eq!(gn.next(), Some(1));
    assert_eq!(gn.next(), Some(2));
    assert_eq!(gn.next(), None);

    //coroutines (i.e. yield returns something useful)
    let mut coro = pin!(
        #[coroutine]
        |_: i32| {
            let x = 1.yield;
            (x + 2).yield;
        }
    );

    assert_eq!(coro.as_mut().resume(0), CoroutineState::Yielded(1));
    assert_eq!(coro.as_mut().resume(2), CoroutineState::Yielded(4));
    assert_eq!(coro.as_mut().resume(3), CoroutineState::Complete(()));
}
