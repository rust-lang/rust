//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass
//@[next] compile-flags: -Znext-solver
#![feature(coroutines, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};

fn foo() -> impl Coroutine<Yield = (), Return = ()> {
    #[coroutine] || {
        let mut gen = Box::pin(foo());
        //[next]~^ ERROR type annotations needed
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
