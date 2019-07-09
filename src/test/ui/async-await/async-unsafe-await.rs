// run-pass

// edition:2018
// aux-build:arc_wake.rs
// aux-build:wake_once.rs

#![feature(async_await, async_unsafe)]

extern crate arc_wake;
extern crate wake_once;

use std::future::Future;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{Context, Poll};
use arc_wake::ArcWake;
use wake_once::wake_and_yield_once;

struct Counter {
    wakes: AtomicUsize,
}

impl ArcWake for Counter {
    fn wake(self: Arc<Self>) {
        Self::wake_by_ref(&self)
    }
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }
}

async unsafe fn unsafe_async_fn(x: u8) -> u8 {
    wake_and_yield_once().await;
    x
}

struct Foo;

trait Bar {
    fn foo() {}
}

impl Foo {
    async fn async_assoc_item(x: u8) -> u8 {
        unsafe {
            unsafe_async_fn(x).await
        }
    }

    async unsafe fn async_unsafe_assoc_item(x: u8) -> u8 {
        unsafe_async_fn(x).await
    }
}

fn test_future_yields_once_then_returns<F, Fut>(f: F)
where
    F: FnOnce(u8) -> Fut,
    Fut: Future<Output = u8>,
{
    let mut fut = Box::pin(f(9));
    let counter = Arc::new(Counter { wakes: AtomicUsize::new(0) });
    let waker = ArcWake::into_waker(counter.clone());
    let mut cx = Context::from_waker(&waker);
    assert_eq!(0, counter.wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&mut cx));
    assert_eq!(1, counter.wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(Poll::Ready(9), fut.as_mut().poll(&mut cx));
}

fn main() {
    macro_rules! test {
        ($($fn_name:expr,)*) => { $(
            test_future_yields_once_then_returns($fn_name);
        )* }
    }

    test! {
        Foo::async_assoc_item,
        |x| {
            async move {
                unsafe { unsafe_async_fn(x).await }
            }
        },
        |x| {
            async move {
                unsafe { Foo::async_unsafe_assoc_item(x).await }
            }
        },
    }
}
