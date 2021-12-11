// issue 65419 - Attempting to run an async fn after completion mentions generators when it should
// be talking about `async fn`s instead. Should also test what happens when it panics.

// run-fail
// needs-unwind
// error-pattern: thread 'main' panicked at '`async fn` resumed after panicking'
// edition:2018
// ignore-wasm no panic or subprocess support
// ignore-emscripten no panic or subprocess support

#![feature(generators, generator_trait)]

use std::panic;

async fn foo() {
    panic!();
}

fn main() {
    let mut future = Box::pin(foo());
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        executor::block_on(future.as_mut());
    }));

    executor::block_on(future.as_mut());
}

mod executor {
    use core::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    };

    pub fn block_on<F: Future>(mut future: F) -> F::Output {
        let mut future = unsafe { Pin::new_unchecked(&mut future) };

        static VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| unimplemented!("clone"),
            |_| unimplemented!("wake"),
            |_| unimplemented!("wake_by_ref"),
            |_| (),
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
        let mut context = Context::from_waker(&waker);

        loop {
            if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
                break val;
            }
        }
    }
}
