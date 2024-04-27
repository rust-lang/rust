// Test that uninhabited saved local doesn't make the entire variant uninhabited.
//@ run-pass
#![allow(unused)]
#![feature(assert_matches)]
#![feature(coroutine_trait)]
#![feature(coroutines, stmt_expr_attributes)]
#![feature(never_type)]
use std::assert_matches::assert_matches;
use std::ops::Coroutine;
use std::ops::CoroutineState;
use std::pin::Pin;

fn conjure<T>() -> T { loop {} }

fn run<T>(x: bool, y: bool) {
    let mut c = #[coroutine] || {
        if x {
            let a : T;
            if y {
                a = conjure::<T>();
            }
            yield ();
        } else {
            let a : T;
            if y {
                a = conjure::<T>();
            }
            yield ();
        }
    };
    assert_matches!(Pin::new(&mut c).resume(()), CoroutineState::Yielded(()));
    assert_matches!(Pin::new(&mut c).resume(()), CoroutineState::Complete(()));
}

fn main() {
    run::<!>(false, false);
}
