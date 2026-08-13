//! Moving an `Unpin` coroutine while a saved local's drop flag is false must
//! not typed-move that local. Regression test for #161026.

//@ run-pass
//@ edition:2024

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

struct DropU8 {
    _r: u8,
}

impl Drop for DropU8 {
    fn drop(&mut self) {}
}

fn main() {
    let mut a = #[coroutine]
    || {
        let _b: DropU8;
        if false {
            _b = DropU8 { _r: 1 };
        }
        yield 4;
    };
    assert!(matches!(Pin::new(&mut a).resume(()), CoroutineState::Yielded(4)));
    let _moved = a;
}
