// issue 65419 - Attempting to run an async fn after completion mentions coroutines when it should
// be talking about `async fn`s instead.

//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ error-pattern: `async fn` resumed after completion
//@ edition:2018

#![feature(coroutines, coroutine_trait)]

async fn foo() {
}

fn main() {
    let mut future = Box::pin(foo());
    executor::block_on(future.as_mut());
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
