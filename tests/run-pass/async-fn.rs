#![feature(
    async_await,
    await_macro,
)]

use std::{future::Future, pin::Pin, task::Poll, ptr};
use std::task::{Waker, RawWaker, RawWakerVTable, Context};

// See if we can run a basic `async fn`
pub async fn foo(x: &u32, y: u32) -> u32 {
    let y = &y;
    let z = 9;
    let z = &z;
    let y = await!(async { *y + *z });
    let a = 10;
    let a = &a;
    *x + y + *a
}

fn raw_waker_clone(_this: *const ()) -> RawWaker {
    panic!("unimplemented");
}
fn raw_waker_wake(_this: *const ()) {
    panic!("unimplemented");
}
fn raw_waker_wake_by_ref(_this: *const ()) {
    panic!("unimplemented");
}
fn raw_waker_drop(_this: *const ()) {}

static RAW_WAKER: RawWakerVTable = RawWakerVTable::new(
    raw_waker_clone,
    raw_waker_wake,
    raw_waker_wake_by_ref,
    raw_waker_drop,
);

fn main() {
    let x = 5;
    let mut fut = foo(&x, 7);
    let waker = unsafe { Waker::from_raw(RawWaker::new(ptr::null(), &RAW_WAKER)) };
    let mut context = Context::from_waker(&waker);
    assert_eq!(unsafe { Pin::new_unchecked(&mut fut) }.poll(&mut context), Poll::Ready(31));
}
