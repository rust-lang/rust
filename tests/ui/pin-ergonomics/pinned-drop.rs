//@ run-pass
//@ edition:2024
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Waker};

#[pin_v2]
struct Foo {
    dropped: Arc<AtomicBool>,
}

impl Foo {
    fn new(dropped: Arc<AtomicBool>) -> Self {
        Self { dropped }
    }
}

impl Drop for Foo {
    fn pin_drop(self: Pin<&mut Self>) {
        self.dropped.store(true, Ordering::Relaxed);
    }
}

impl Future for Foo {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

fn block_on<T, F: Future<Output = T>>(mut f: F) -> T {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);

    let f = &pin mut f;
    loop {
        if let Poll::Ready(ret) = f.poll(&mut cx) {
            break ret;
        }
    }
}

fn main() {
    let dropped = Arc::new(AtomicBool::new(false));
    block_on(Foo::new(dropped.clone()));
    assert!(dropped.load(Ordering::Relaxed));
}
