//@revisions: stack tree
//@compile-flags: -Zmiri-strict-provenance
//@[tree]compile-flags: -Zmiri-tree-borrows

// WARNING: If you would ever want to modify this test,
// please consider modifying rustc's async drop test at
// `tests/ui/async-await/async-drop.rs`.

#![feature(async_drop, impl_trait_in_assoc_type, noop_waker, async_closure)]
#![allow(incomplete_features, dead_code)]

// FIXME(zetanumbers): consider AsyncDestruct::async_drop cleanup tests
use core::future::{AsyncDrop, Future, async_drop_in_place};
use core::hint::black_box;
use core::mem::{self, ManuallyDrop};
use core::pin::{Pin, pin};
use core::task::{Context, Poll, Waker};

async fn test_async_drop<T>(x: T) {
    let mut x = mem::MaybeUninit::new(x);
    let dtor = pin!(unsafe { async_drop_in_place(x.as_mut_ptr()) });
    test_idempotency(dtor).await;
}

fn test_idempotency<T>(mut x: Pin<&mut T>) -> impl Future<Output = ()> + '_
where
    T: Future<Output = ()>,
{
    core::future::poll_fn(move |cx| {
        assert_eq!(x.as_mut().poll(cx), Poll::Ready(()));
        assert_eq!(x.as_mut().poll(cx), Poll::Ready(()));
        Poll::Ready(())
    })
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
        test_async_drop(|| {
            black_box(foo);
            let foo = AsyncInt(10);
            foo
        })
        .await;

        test_async_drop(AsyncEnum::A(AsyncInt(12))).await;
        test_async_drop(AsyncEnum::B(SyncInt(13))).await;

        test_async_drop(SyncInt(14)).await;
        test_async_drop(SyncThenAsync { i: 15, a: AsyncInt(16), b: SyncInt(17), c: AsyncInt(18) })
            .await;

        let async_drop_fut = pin!(core::future::async_drop(AsyncInt(19)));
        test_idempotency(async_drop_fut).await;

        let foo = AsyncInt(20);
        test_async_drop(async || {
            black_box(foo);
            let foo = AsyncInt(19);
            // Await point there, but this is async closure so it's fine
            black_box(core::future::ready(())).await;
            foo
        })
        .await;

        test_async_drop(AsyncUnion { signed: 21 }).await;
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
    type Dropper<'a>
        = impl Future<Output = ()>
    where
        Self: 'a;

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

// FIXME(zetanumbers): Disallow types with `AsyncDrop` in unions
union AsyncUnion {
    signed: i32,
    unsigned: u32,
}

impl AsyncDrop for AsyncUnion {
    type Dropper<'a> = impl Future<Output = ()>;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!("AsyncUnion::Dropper::poll: {}, {}", unsafe { self.signed }, unsafe {
                self.unsigned
            });
        }
    }
}
