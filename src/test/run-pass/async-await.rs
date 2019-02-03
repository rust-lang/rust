// edition:2018

#![feature(arbitrary_self_types, async_await, await_macro, futures_api)]

use std::pin::Pin;
use std::future::Future;
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

struct WakeOnceThenComplete(bool);

fn wake_and_yield_once() -> WakeOnceThenComplete { WakeOnceThenComplete(false) }

impl Future for WakeOnceThenComplete {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, waker: &Waker) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            waker.wake();
            self.0 = true;
            Poll::Pending
        }
    }
}

fn async_block(x: u8) -> impl Future<Output = u8> {
    async move {
        await!(wake_and_yield_once());
        x
    }
}

fn async_block_with_borrow_named_lifetime<'a>(x: &'a u8) -> impl Future<Output = u8> + 'a {
    async move {
        await!(wake_and_yield_once());
        *x
    }
}

fn async_nonmove_block(x: u8) -> impl Future<Output = u8> {
    async move {
        let future = async {
            await!(wake_and_yield_once());
            x
        };
        await!(future)
    }
}

fn async_closure(x: u8) -> impl Future<Output = u8> {
    (async move |x: u8| -> u8 {
        await!(wake_and_yield_once());
        x
    })(x)
}

async fn async_fn(x: u8) -> u8 {
    await!(wake_and_yield_once());
    x
}

async fn async_fn_with_borrow(x: &u8) -> u8 {
    await!(wake_and_yield_once());
    *x
}

async fn async_fn_with_borrow_named_lifetime<'a>(x: &'a u8) -> u8 {
    await!(wake_and_yield_once());
    *x
}

fn async_fn_with_impl_future_named_lifetime<'a>(x: &'a u8) -> impl Future<Output = u8> + 'a {
    async move {
        await!(wake_and_yield_once());
        *x
    }
}

async fn async_fn_with_named_lifetime_multiple_args<'a>(x: &'a u8, _y: &'a u8) -> u8 {
    await!(wake_and_yield_once());
    *x
}

fn async_fn_with_internal_borrow(y: u8) -> impl Future<Output = u8> {
    async move {
        await!(async_fn_with_borrow(&y))
    }
}

unsafe async fn unsafe_async_fn(x: u8) -> u8 {
    await!(wake_and_yield_once());
    x
}

struct Foo;

trait Bar {
    fn foo() {}
}

impl Foo {
    async fn async_method(x: u8) -> u8 {
        unsafe {
            await!(unsafe_async_fn(x))
        }
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
    assert_eq!(0, counter.wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(Poll::Pending, fut.as_mut().poll(&waker));
    assert_eq!(1, counter.wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(Poll::Ready(9), fut.as_mut().poll(&waker));
}

fn main() {
    macro_rules! test {
        ($($fn_name:expr,)*) => { $(
            test_future_yields_once_then_returns($fn_name);
        )* }
    }

    macro_rules! test_with_borrow {
        ($($fn_name:expr,)*) => { $(
            test_future_yields_once_then_returns(|x| {
                async move {
                    await!($fn_name(&x))
                }
            });
        )* }
    }

    test! {
        async_block,
        async_nonmove_block,
        async_closure,
        async_fn,
        async_fn_with_internal_borrow,
        Foo::async_method,
        |x| {
            async move {
                unsafe { await!(unsafe_async_fn(x)) }
            }
        },
    }

    test_with_borrow! {
        async_block_with_borrow_named_lifetime,
        async_fn_with_borrow,
        async_fn_with_borrow_named_lifetime,
        async_fn_with_impl_future_named_lifetime,
        |x| {
            async move {
                await!(async_fn_with_named_lifetime_multiple_args(x, x))
            }
        },
    }
}
