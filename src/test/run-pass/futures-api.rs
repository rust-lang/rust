// aux-build:arc_wake.rs

#![feature(arbitrary_self_types, futures_api)]
#![allow(unused)]

extern crate arc_wake;

use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{
    Poll, Waker,
};
use arc_wake::ArcWake;

struct Counter {
    wakes: AtomicUsize,
}

impl ArcWake for Counter {
    fn wake(arc_self: &Arc<Self>) {
        arc_self.wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }
}

struct MyFuture;

impl Future for MyFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, waker: &Waker) -> Poll<Self::Output> {
        // Wake twice
        waker.wake();
        waker.wake();
        Poll::Ready(())
    }
}

fn test_waker() {
    let counter = Arc::new(Counter {
        wakes: AtomicUsize::new(0),
    });
    let waker = ArcWake::into_waker(counter.clone());
    assert_eq!(2, Arc::strong_count(&counter));

    assert_eq!(Poll::Ready(()), Pin::new(&mut MyFuture).poll(&waker));
    assert_eq!(2, counter.wakes.load(atomic::Ordering::SeqCst));

    drop(waker);
    assert_eq!(1, Arc::strong_count(&counter));
}

fn main() {
    test_waker();
}
