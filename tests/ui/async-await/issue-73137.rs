// Regression test for <https://github.com/rust-lang/rust/issues/73137>

// run-pass
// edition:2018
// revisions: normal drop-tracking
// [normal]compile-flags: -Zdrop-tracking=no
// [drop-tracking]compile-flags: -Zdrop-tracking

#![allow(dead_code)]
use std::future::Future;
use std::task::{Waker, Wake, Context};
use std::sync::Arc;

struct DummyWaker;
impl Wake for DummyWaker {
    fn wake(self: Arc<Self>) {}
}

struct Foo {
    a: usize,
    b: &'static u32,
}

#[inline(never)]
fn nop<T>(_: T) {}

fn main() {
    let mut fut = Box::pin(async {
        let action = Foo {
            b: &42,
            a: async { 0 }.await,
        };

        // An error in the generator transform caused `b` to be overwritten with `a` when `b` was
        // borrowed.
        nop(&action.b);
        assert_ne!(0usize, unsafe { std::mem::transmute(action.b) });

        async {}.await;
    });
    let waker = Waker::from(Arc::new(DummyWaker));
    let mut cx = Context::from_waker(&waker);
    let _ = fut.as_mut().poll(&mut cx);
}
