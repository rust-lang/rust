//@ run-pass
//@ check-run-results

// WARNING: If you would ever want to modify this test,
// please consider modifying miri's async drop test at
// `src/tools/miri/tests/pass/async-drop.rs`.

#![feature(async_drop, impl_trait_in_assoc_type)]
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
        test_async_drop(Int(0), 16).await;
        test_async_drop(AsyncInt(0), 32).await;
        test_async_drop([AsyncInt(1), AsyncInt(2)], 104).await;
        test_async_drop((AsyncInt(3), AsyncInt(4)), 120).await;
        test_async_drop(5, 16).await;
        let j = 42;
        test_async_drop(&i, 16).await;
        test_async_drop(&j, 16).await;
        test_async_drop(
            AsyncStruct { b: AsyncInt(8), a: AsyncInt(7), i: 6 },
            136,
        ).await;
        test_async_drop(ManuallyDrop::new(AsyncInt(9)), 16).await;

        let foo = AsyncInt(10);
        test_async_drop(AsyncReference { foo: &foo }, 32).await;
        let _ = ManuallyDrop::new(foo);

        let foo = AsyncInt(11);
        test_async_drop(
            || {
                black_box(foo);
                let foo = AsyncInt(10);
                foo
            },
            48,
        )
        .await;

        test_async_drop(AsyncEnum::A(AsyncInt(12)), 104).await;
        test_async_drop(AsyncEnum::B(SyncInt(13)), 104).await;

        test_async_drop(SyncInt(14), 16).await;
        test_async_drop(
            SyncThenAsync { i: 15, a: AsyncInt(16), b: SyncInt(17), c: AsyncInt(18) },
            120,
        )
        .await;

        let mut ptr19 = mem::MaybeUninit::new(AsyncInt(19));
        let async_drop_fut = pin!(unsafe { async_drop_in_place(ptr19.as_mut_ptr()) });
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
            48,
        )
        .await;

        test_async_drop(AsyncUnion { signed: 21 }, 32).await;
    });
    let res = fut.poll(&mut cx);
    assert_eq!(res, Poll::Ready(()));
}

struct AsyncInt(i32);

impl Drop for AsyncInt {
    fn drop(&mut self) {
        println!("AsyncInt::drop: {}", self.0);
    }
}
impl AsyncDrop for AsyncInt {
    async fn drop(self: Pin<&mut Self>) {
        println!("AsyncInt::Dropper::poll: {}", self.0);
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

impl Drop for AsyncReference<'_> {
    fn drop(&mut self) {
        println!("AsyncReference::drop: {}", self.foo.0);
    }
}

impl AsyncDrop for AsyncReference<'_> {
    async fn drop(self: Pin<&mut Self>) {
        println!("AsyncReference::Dropper::poll: {}", self.foo.0);
    }
}

struct Int(i32);

struct AsyncStruct {
    i: i32,
    a: AsyncInt,
    b: AsyncInt,
}

impl Drop for AsyncStruct {
    fn drop(&mut self) {
        println!("AsyncStruct::drop: {}", self.i);
    }
}

impl AsyncDrop for AsyncStruct {
    async fn drop(self: Pin<&mut Self>) {
        println!("AsyncStruct::Dropper::poll: {}", self.i);
    }
}

enum AsyncEnum {
    A(AsyncInt),
    B(SyncInt),
}

impl Drop for AsyncEnum {
    fn drop(&mut self) {
        let new_self = match self {
            AsyncEnum::A(foo) => {
                println!("AsyncEnum(A)::drop: {}", foo.0);
                AsyncEnum::B(SyncInt(foo.0))
            }
            AsyncEnum::B(foo) => {
                println!("AsyncEnum(B)::drop: {}", foo.0);
                AsyncEnum::A(AsyncInt(foo.0))
            }
        };
        mem::forget(mem::replace(&mut *self, new_self));
    }
}
impl AsyncDrop for AsyncEnum {
    async fn drop(mut self: Pin<&mut Self>) {
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

// FIXME(zetanumbers): Disallow types with `AsyncDrop` in unions
union AsyncUnion {
    signed: i32,
    unsigned: u32,
}

impl Drop for AsyncUnion {
    fn drop(&mut self) {
        println!(
            "AsyncUnion::drop: {}, {}",
            unsafe { self.signed },
            unsafe { self.unsigned },
        );
    }
}
impl AsyncDrop for AsyncUnion {
    async fn drop(self: Pin<&mut Self>) {
        println!(
            "AsyncUnion::Dropper::poll: {}, {}",
            unsafe { self.signed },
            unsafe { self.unsigned },
        );
    }
}
