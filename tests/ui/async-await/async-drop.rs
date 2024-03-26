//@ run-pass
//@ check-run-results

#![feature(async_drop, impl_trait_in_assoc_type, noop_waker)]
#![allow(incomplete_features, dead_code)]

//@ edition: 2021

use core::future::{async_drop_in_place, poll_fn, AsyncDrop, Future};
use core::hint::black_box;
use core::mem::{self, ManuallyDrop};
use core::pin::{pin, Pin};
use core::task::{Context, Poll, Waker};

// TODO: Check idempotency of async_drop
async fn test_async_drop<T>(x: T) {
    let mut x = mem::MaybeUninit::new(x);
    let mut dtor = pin!(unsafe { async_drop_in_place(x.as_mut_ptr()) });
    poll_fn(|cx| {
        assert_eq!(dtor.as_mut().poll(cx), Poll::Ready(()));
        // Check for idempotency
        assert_eq!(dtor.as_mut().poll(cx), Poll::Ready(()));
        Poll::Ready(())
    }).await;
}

fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);

    let i = 13;
    let fut = pin!(async {
        test_async_drop(Int(0)).await;
        test_async_drop(AsyncInt(0)).await;
        test_async_drop([AsyncInt(1), AsyncInt(2)]).await;
        test_async_drop((AsyncInt(3), AsyncInt(4))).await;
        test_async_drop(5).await;
        let j = 42;
        test_async_drop(&i).await;
        test_async_drop(&j).await;
        test_async_drop(AsyncStruct { b: AsyncInt(8), a: AsyncInt(7), i: 6 }).await;
        test_async_drop(ManuallyDrop::new(AsyncInt(9))).await;

        let foo = AsyncInt(10);
        test_async_drop(AsyncReference { foo: &foo }).await;

        let foo = AsyncInt(11);
        test_async_drop(|| black_box(foo)).await;

        test_async_drop(AsyncEnum::A(AsyncInt(12))).await;
        test_async_drop(AsyncEnum::B(SyncInt(13))).await;

        test_async_drop(SyncThenAsync {
            i: 14,
            a: AsyncInt(15),
            b: SyncInt(16),
            c: AsyncInt(17),
        }).await;
    });
    let res = fut.poll(&mut cx);
    assert_eq!(res, Poll::Ready(()));
}

struct AsyncInt(i32);

impl AsyncDrop for AsyncInt {
    type Dropper<'a> = impl Future<Output = ()>;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!("AsyncInt::Dropper::poll: {}", self.0);
        }
    }
}

struct SyncInt(i32);

impl Drop for SyncInt {
    fn drop(&mut self) {
        println!("SyncInt::drop: {}", self.0);
    }
}

struct SyncThenAsync {
    i: i32,
    a: AsyncInt,
    b: SyncInt,
    c: AsyncInt,
}

impl Drop for SyncThenAsync {
    fn drop(&mut self) {
        println!("SyncThenAsync::drop: {}", self.i);
    }
}

struct AsyncReference<'a> {
    foo: &'a AsyncInt,
}

impl AsyncDrop for AsyncReference<'_> {
    type Dropper<'a> = impl Future<Output = ()> where Self: 'a;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!("AsyncReference::Dropper::poll: {}", self.foo.0);
        }
    }
}

struct Int(i32);

struct AsyncStruct {
    i: i32,
    a: AsyncInt,
    b: AsyncInt,
}

impl AsyncDrop for AsyncStruct {
    type Dropper<'a> = impl Future<Output = ()>;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!("AsyncStruct::Dropper::poll: {}", self.i);
        }
    }
}

enum AsyncEnum {
    A(AsyncInt),
    B(SyncInt),
}

impl AsyncDrop for AsyncEnum {
    type Dropper<'a> = impl Future<Output = ()>;

    fn async_drop(mut self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            let new_self = match &*self {
                AsyncEnum::A(foo) => {
                    println!("AsyncEnum(A)::Dropper::poll: {}", foo.0);
                    AsyncEnum::B(SyncInt(foo.0))
                }
                AsyncEnum::B(foo) => {
                    println!("AsyncEnum(B)::Dropper::poll: {}", foo.0);
                    AsyncEnum::A(AsyncInt(foo.0))
                }
            };
            mem::forget(mem::replace(&mut *self, new_self));
        }
    }
}
