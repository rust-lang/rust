//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(coroutines, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};

fn foo() -> impl Coroutine<Yield = (), Return = ()> {
    #[coroutine] || {
        let mut gen = Box::pin(foo());
        let mut r = gen.as_mut().resume(());
        while let CoroutineState::Yielded(v) = r {
            yield v;
            r = gen.as_mut().resume(());
        }
    }
}

fn main() {
    foo();
}
