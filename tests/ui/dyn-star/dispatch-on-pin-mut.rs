// run-pass
// edition:2021
// check-run-results

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

use std::future::Future;

async fn foo(f: dyn* Future<Output = i32>) {
    println!("value: {}", f.await);
}

async fn async_main() {
    foo(Box::pin(async { 1 })).await
}

// ------------------------------------------------------------------------- //
// Implementation Details Below...

use std::pin::Pin;
use std::task::*;

pub fn noop_waker() -> Waker {
    let raw = RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE);

    // SAFETY: the contracts for RawWaker and RawWakerVTable are upheld
    unsafe { Waker::from_raw(raw) }
}

const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

unsafe fn noop_clone(_p: *const ()) -> RawWaker {
    RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE)
}

unsafe fn noop(_p: *const ()) {}

fn main() {
    let mut fut = async_main();

    // Poll loop, just to test the future...
    let waker = noop_waker();
    let ctx = &mut Context::from_waker(&waker);

    loop {
        match unsafe { Pin::new_unchecked(&mut fut).poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(()) => break,
        }
    }
}
