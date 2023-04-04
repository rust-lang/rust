// edition: 2021
// known-bug: #108309
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![feature(min_specialization)]

struct MyStruct;

trait MyTrait<T> {
    async fn foo(_: T) -> &'static str;
}

impl<T> MyTrait<T> for MyStruct {
    default async fn foo(_: T) -> &'static str {
        "default"
    }
}

impl MyTrait<i32> for MyStruct {
    async fn foo(_: i32) -> &'static str {
        "specialized"
    }
}

async fn async_main() {
    assert_eq!(MyStruct::foo(42).await, "specialized");
    assert_eq!(indirection(42).await, "specialized");
}

async fn indirection<T>(x: T) -> &'static str {
    //explicit type coercion is currently necessary
    // because of https://github.com/rust-lang/rust/issues/67918
    <MyStruct as MyTrait<T>>::foo(x).await
}

// ------------------------------------------------------------------------- //
// Implementation Details Below...

use std::future::Future;
use std::pin::Pin;
use std::task::*;

pub fn noop_waker() -> Waker {
    let raw = RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE);

    // SAFETY: the contracts for RawWaker and RawWakerVTable are upheld
    unsafe { Waker::from_raw(raw) }
}

const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

unsafe fn noop_clone(_p: *const ()) -> RawWaker {
    RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE)
}

unsafe fn noop(_p: *const ()) {}

fn main() {
    let mut fut = async_main();

    // Poll loop, just to test the future...
    let waker = noop_waker();
    let ctx = &mut Context::from_waker(&waker);

    loop {
        match unsafe { Pin::new_unchecked(&mut fut).poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(()) => break,
        }
    }
}
