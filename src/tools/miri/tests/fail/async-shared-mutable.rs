//! FIXME: This test should pass! However, `async fn` does not yet use `UnsafePinned`.
//! This is a regression test for <https://github.com/rust-lang/rust/issues/137750>:
//! `UnsafePinned` must include the effects of `UnsafeCell`.
//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@normalize-stderr-test: "\[0x[a-fx\d.]+\]" -> "[OFFSET]"

use core::future::Future;
use core::pin::{Pin, pin};
use core::task::{Context, Poll, Waker};

fn main() {
    let mut f = pin!(async move {
        let x = &mut 0u8;
        core::future::poll_fn(move |_| {
            *x = 1; //~ERROR: write access
            Poll::<()>::Pending
        })
        .await
    });
    let mut cx = Context::from_waker(&Waker::noop());
    assert_eq!(f.as_mut().poll(&mut cx), Poll::Pending);
    let _: Pin<&_> = f.as_ref(); // Or: `f.as_mut().into_ref()`.
    assert_eq!(f.as_mut().poll(&mut cx), Poll::Pending);
}
