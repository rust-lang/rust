//@ revisions: current next
//@[current] check-pass
//@[next] compile-flags: -Znext-solver
#![feature(coroutines, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};

fn foo() -> impl Coroutine<Yield = (), Return = ()> {
    || {
        let mut gen = Box::pin(foo());
        //[next]~^ ERROR type annotations needed
        //[next]~| ERROR type annotations needed
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
