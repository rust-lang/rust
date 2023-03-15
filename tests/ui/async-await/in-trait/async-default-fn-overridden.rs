// run-pass
// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
//~^ WARN the feature `async_fn_in_trait` is incomplete and may not be safe to use

use std::future::Future;

trait AsyncTrait {
    async fn default_impl() {
        assert!(false);
    }

    async fn call_default_impl() {
        Self::default_impl().await
    }
}

struct AsyncType;

impl AsyncTrait for AsyncType {
    async fn default_impl() {
        // :)
    }
}

async fn async_main() {
    // Should not assert false
    AsyncType::call_default_impl().await;
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
