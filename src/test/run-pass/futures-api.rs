#![feature(arbitrary_self_types, futures_api)]
#![allow(unused)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{
    Poll, Waker, RawWaker, RawWakerVTable,
};

macro_rules! waker_vtable {
    ($ty:ident) => {
        &RawWakerVTable {
            clone: clone_arc_raw::<$ty>,
            drop: drop_arc_raw::<$ty>,
            wake: wake_arc_raw::<$ty>,
        }
    };
}

pub trait ArcWake {
    fn wake(arc_self: &Arc<Self>);

    fn into_waker(wake: Arc<Self>) -> Waker where Self: Sized
    {
        let ptr = Arc::into_raw(wake) as *const();

        unsafe {
            Waker::new_unchecked(RawWaker{
                data: ptr,
                vtable: waker_vtable!(Self),
            })
        }
    }
}

unsafe fn increase_refcount<T: ArcWake>(data: *const()) {
    // Retain Arc by creating a copy
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    let arc_clone = arc.clone();
    // Forget the Arcs again, so that the refcount isn't decrased
    let _ = Arc::into_raw(arc);
    let _ = Arc::into_raw(arc_clone);
}

unsafe fn clone_arc_raw<T: ArcWake>(data: *const()) -> RawWaker {
    increase_refcount::<T>(data);
    RawWaker {
        data: data,
        vtable: waker_vtable!(T),
    }
}

unsafe fn drop_arc_raw<T: ArcWake>(data: *const()) {
    // Drop Arc
    let _: Arc<T> = Arc::from_raw(data as *const T);
}

unsafe fn wake_arc_raw<T: ArcWake>(data: *const()) {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    ArcWake::wake(&arc);
    let _ = Arc::into_raw(arc);
}

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
