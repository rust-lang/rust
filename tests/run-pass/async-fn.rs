// ignore-test FIXME ignored to let https://github.com/rust-lang/rust/pull/59119 land
#![feature(
    async_await,
    await_macro,
    futures_api,
)]

use std::{future::Future, pin::Pin, task::Poll, ptr};
use std::task::{Waker, RawWaker, RawWakerVTable};

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
fn raw_waker_drop(_this: *const ()) {}

static RAW_WAKER: RawWakerVTable = RawWakerVTable {
    clone: raw_waker_clone,
    wake: raw_waker_wake,
    drop: raw_waker_drop,
};

fn main() {
    let x = 5;
    let mut fut = foo(&x, 7);
    let waker = unsafe { Waker::new_unchecked(RawWaker::new(ptr::null(), &RAW_WAKER)) };
    assert_eq!(unsafe { Pin::new_unchecked(&mut fut) }.poll(&waker), Poll::Ready(31));
}
