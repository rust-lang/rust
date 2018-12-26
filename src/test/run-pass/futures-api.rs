#![feature(arbitrary_self_types, futures_api)]
#![allow(unused)]

use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{
    Poll, Wake, Waker, LocalWaker,
    local_waker, local_waker_from_nonlocal,
};

struct Counter {
    local_wakes: AtomicUsize,
    nonlocal_wakes: AtomicUsize,
}

impl Wake for Counter {
    fn wake(this: &Arc<Self>) {
        this.nonlocal_wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }

    unsafe fn wake_local(this: &Arc<Self>) {
        this.local_wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }
}

struct MyFuture;

impl Future for MyFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, lw: &LocalWaker) -> Poll<Self::Output> {
        // Wake once locally
        lw.wake();
        // Wake twice non-locally
        let waker = lw.clone().into_waker();
        waker.wake();
        waker.wake();
        Poll::Ready(())
    }
}

fn test_local_waker() {
    let counter = Arc::new(Counter {
        local_wakes: AtomicUsize::new(0),
        nonlocal_wakes: AtomicUsize::new(0),
    });
    let waker = unsafe { local_waker(counter.clone()) };
    assert_eq!(Poll::Ready(()), Pin::new(&mut MyFuture).poll(&waker));
    assert_eq!(1, counter.local_wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(2, counter.nonlocal_wakes.load(atomic::Ordering::SeqCst));
}

fn test_local_as_nonlocal_waker() {
    let counter = Arc::new(Counter {
        local_wakes: AtomicUsize::new(0),
        nonlocal_wakes: AtomicUsize::new(0),
    });
    let waker: LocalWaker = local_waker_from_nonlocal(counter.clone());
    assert_eq!(Poll::Ready(()), Pin::new(&mut MyFuture).poll(&waker));
    assert_eq!(0, counter.local_wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(3, counter.nonlocal_wakes.load(atomic::Ordering::SeqCst));
}

fn main() {
    test_local_waker();
    test_local_as_nonlocal_waker();
}
