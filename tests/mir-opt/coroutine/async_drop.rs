//@ skip-filecheck
//@ compile-flags: -Zmir-opt-level=0
//@ needs-unwind
//@ edition: 2024

// WARNING: If you would ever want to modify this test,
// please consider modifying rustc's async drop test at
// `tests/ui/async-await/async-drop/async-drop-initial.rs`.

#![feature(async_drop, impl_trait_in_assoc_type)]
#![allow(incomplete_features, dead_code, unused_variables)]

// FIXME(zetanumbers): consider AsyncDestruct::async_drop cleanup tests
use core::future::{AsyncDrop, Future, async_drop_in_place};
use core::hint::black_box;
use core::mem::{self, ManuallyDrop};
use core::pin::{Pin, pin};
use core::task::{Context, Poll, Waker};

async fn test_async_drop<T>(x: T) {
    let mut x = mem::MaybeUninit::new(x);
    let dtor = pin!(unsafe { async_drop_in_place(&mut *x.as_mut_ptr()) });
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

// EMIT_MIR async_drop.simple-{closure#0}.ElaborateDrops.diff
// EMIT_MIR async_drop.simple-{closure#0}.StateTransform.diff
// EMIT_MIR async_drop.simple-{closure#0}.coroutine_drop_async.0.mir
async fn simple() {
    let sync_int = SyncInt(0);
    let async_int = AsyncInt(0);
}

// EMIT_MIR async_drop.double-{closure#0}.ElaborateDrops.diff
// EMIT_MIR async_drop.double-{closure#0}.StateTransform.diff
// EMIT_MIR async_drop.double-{closure#0}.coroutine_drop_async.0.mir
async fn double() {
    let sync_int = SyncInt(0);
    let async_int = AsyncInt(0);
    let async_int_again = AsyncInt(0);
}

// EMIT_MIR async_drop.elaborate_drops-{closure#0}.ElaborateDrops.diff
// EMIT_MIR async_drop.elaborate_drops-{closure#0}.StateTransform.diff
async fn elaborate_drops() {
    let sync_int = SyncInt(0);
    let async_int = AsyncInt(0);
    let tuple = [AsyncInt(1), AsyncInt(2)];

    let async_struct = AsyncStruct { b: AsyncInt(5), a: AsyncInt(4), i: 3 };
    let async_struct_mix = SyncThenAsync { i: 6, a: AsyncInt(7), b: SyncInt(8), c: AsyncInt(9) };
    let async_enum = AsyncEnum::A(AsyncInt(10));

    let manually_drop_async_int = ManuallyDrop::new(AsyncInt(11));
    let foo = AsyncInt(12);
    let async_ref = AsyncReference { foo: &foo };

    let foo = AsyncInt(14);
    let async_closure = || {
        black_box(foo);
        let foo = AsyncInt(13);
        foo
    };

    // We test dropping the coroutine, not running it.
    let foo = AsyncInt(15);
    let async_coroutine = async || {
        black_box(foo);
        let foo = AsyncInt(16);
        // Await point there, but this is async closure so it's fine
        black_box(core::future::ready(())).await;
        foo
    };
}

fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);

    let i = 13;
    let fut = pin!(async {
        elaborate_drops().await;

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

        let mut ptr19 = mem::MaybeUninit::new(AsyncInt(19));
        let async_drop_fut = pin!(unsafe { async_drop_in_place(&mut *ptr19.as_mut_ptr()) });
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
    });
    let res = fut.poll(&mut cx);
    assert_eq!(res, Poll::Ready(()));
}

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncInt.make_shim.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncInt.StateTransform.diff
struct AsyncInt(i32);

impl Drop for AsyncInt {
    fn drop(&mut self) {
        println!("AsyncInt::drop: {}", self.0);
    }
}
impl AsyncDrop for AsyncInt {
    async fn drop(self: Pin<&mut Self>) {
        println!("AsyncInt::async_drop: {}", self.0);
    }
}

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.SyncInt.make_shim.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.SyncInt.StateTransform.diff
struct SyncInt(i32);

impl Drop for SyncInt {
    fn drop(&mut self) {
        println!("SyncInt::drop: {}", self.0);
    }
}

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.SyncThenAsync.make_shim.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.SyncThenAsync.StateTransform.diff
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

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncReference_'__.make_shim.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncReference_'__.StateTransform.diff
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
        println!("AsyncReference::async_drop: {}", self.foo.0);
    }
}

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.Int.make_shim.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.Int.StateTransform.diff
struct Int(i32);

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncStruct.make_shim.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncStruct.StateTransform.diff
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
        println!("AsyncStruct::async_drop: {}", self.i);
    }
}

// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncEnum.make_shim.0.mir
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.AsyncEnum.StateTransform.diff
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
                println!("AsyncEnum(A)::async_drop: {}", foo.0);
                AsyncEnum::B(SyncInt(foo.0))
            }
            AsyncEnum::B(foo) => {
                println!("AsyncEnum(B)::async_drop: {}", foo.0);
                AsyncEnum::A(AsyncInt(foo.0))
            }
        };
        mem::forget(mem::replace(&mut *self, new_self));
    }
}
