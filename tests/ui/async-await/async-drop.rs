//@ run-pass
//@ check-run-results

// WARNING: If you would ever want to modify this test,
// please consider modifying miri's async drop test at
// `src/tools/miri/tests/pass/async-drop.rs`.

#![feature(async_drop, impl_trait_in_assoc_type, noop_waker, async_closure)]
#![allow(incomplete_features, dead_code)]

//@ edition: 2021

// FIXME(zetanumbers): consider AsyncDestruct::async_drop cleanup tests
use core::future::{async_drop_in_place, AsyncDrop, Future};
use core::hint::black_box;
use core::mem::{self, ManuallyDrop};
use core::pin::{pin, Pin};
use core::task::{Context, Poll, Waker};

async fn test_async_drop<T>(x: T, _size: usize) {
    let mut x = mem::MaybeUninit::new(x);
    let dtor = pin!(unsafe { async_drop_in_place(x.as_mut_ptr()) });

    // FIXME(zetanumbers): This check fully depends on the layout of
    // the coroutine state, since async destructor combinators are just
    // async functions.
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        mem::size_of_val(&*dtor),
        _size,
        "sizes did not match for async destructor of type {}",
        core::any::type_name::<T>(),
    );

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
        test_async_drop(Int(0), 0).await;
        // FIXME(#63818): niches in coroutines are disabled.
        // Some of these sizes should be smaller, as indicated in comments.
        test_async_drop(AsyncInt(0), /*104*/ 112).await;
        test_async_drop([AsyncInt(1), AsyncInt(2)], /*152*/ 168).await;
        test_async_drop((AsyncInt(3), AsyncInt(4)), /*488*/ 528).await;
        test_async_drop(5, 0).await;
        let j = 42;
        test_async_drop(&i, 0).await;
        test_async_drop(&j, 0).await;
        test_async_drop(AsyncStruct { b: AsyncInt(8), a: AsyncInt(7), i: 6 }, /*1688*/ 1792).await;
        test_async_drop(ManuallyDrop::new(AsyncInt(9)), 0).await;

        let foo = AsyncInt(10);
        test_async_drop(AsyncReference { foo: &foo }, /*104*/ 112).await;

        let foo = AsyncInt(11);
        test_async_drop(
            || {
                black_box(foo);
                let foo = AsyncInt(10);
                foo
            },
            /*120*/ 136,
        )
        .await;

        test_async_drop(AsyncEnum::A(AsyncInt(12)), /*680*/ 736).await;
        test_async_drop(AsyncEnum::B(SyncInt(13)), /*680*/ 736).await;

        test_async_drop(SyncInt(14), /*16*/ 24).await;
        test_async_drop(
            SyncThenAsync { i: 15, a: AsyncInt(16), b: SyncInt(17), c: AsyncInt(18) },
            /*3064*/ 3296,
        )
        .await;

        let async_drop_fut = pin!(core::future::async_drop(AsyncInt(19)));
        test_idempotency(async_drop_fut).await;

        let foo = AsyncInt(20);
        test_async_drop(
            async || {
                black_box(foo);
                let foo = AsyncInt(19);
                // Await point there, but this is async closure so it's fine
                black_box(core::future::ready(())).await;
                foo
            },
            /*120*/ 136,
        )
        .await;

        test_async_drop(AsyncUnion { signed: 21 }, /*32*/ 40).await;
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

// FIXME(zetanumbers): Disallow types with `AsyncDrop` in unions
union AsyncUnion {
    signed: i32,
    unsigned: u32,
}

impl AsyncDrop for AsyncUnion {
    type Dropper<'a> = impl Future<Output = ()>;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!(
                "AsyncUnion::Dropper::poll: {}, {}",
                unsafe { self.signed },
                unsafe { self.unsigned },
            );
        }
    }
}
