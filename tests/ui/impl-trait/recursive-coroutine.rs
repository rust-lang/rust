// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
#![feature(coroutines, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};

fn foo() -> impl Coroutine<Yield = (), Return = ()> {
    //~^ ERROR cannot resolve opaque type
    //~| NOTE recursive opaque type
    //~| NOTE in this expansion of desugaring of
    || {
        let mut gen = Box::pin(foo());
        //~^ NOTE coroutine captures itself here
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
