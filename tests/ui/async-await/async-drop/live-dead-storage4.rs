// ex-ice: #141409
//@ compile-flags: -Zmir-enable-passes=+Inline -Zvalidate-mir -Zlint-mir --crate-type lib
//@ edition:2024
//@ check-pass

#![feature(async_drop)]
#![allow(incomplete_features)]
#![allow(non_snake_case)]

use std::mem::ManuallyDrop;
use std::{
    future::{async_drop_in_place, Future},
    pin::{pin, Pin},
    sync::{mpsc, Arc},
    task::{Context, Poll, Wake, Waker},
};
fn main() {
    block_on(bar(0))
}
async fn baz(ident_base: usize) {}
async fn bar(ident_base: usize) {
    baz(1).await
}
fn block_on<F>(fut_unpin: F) -> F::Output
where
    F: Future,
{
    let fut_pin = pin!(ManuallyDrop::new(fut_unpin));
    let mut fut = unsafe { Pin::map_unchecked_mut(fut_pin, |x| &mut **x) };
    let (waker, rx) = simple_waker();
    let mut context = Context::from_waker(&waker);
    let rv = loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Ready(out) => break out,
            PollPending => (),
        }
    };
    let drop_fut_unpin = unsafe { async_drop_in_place(fut.get_unchecked_mut()) };
    let drop_fut = pin!(drop_fut_unpin);
    loop {
        match drop_fut.poll(&mut context) {
            Poll => break,
        }
    }
    rv
}
fn simple_waker() -> (Waker, mpsc::Receiver<()>) {
    struct SimpleWaker {
        tx: mpsc::Sender<()>,
    }
    impl Wake for SimpleWaker {
        fn wake(self: Arc<Self>) {}
    }
    let (tx, rx) = mpsc::channel();
    (Waker::from(Arc::new(SimpleWaker { tx })), rx)
}
