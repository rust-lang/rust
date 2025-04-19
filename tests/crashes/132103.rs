//@ known-bug: #132103
//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes
//@ edition: 2018
use core::future::{async_drop_in_place, Future};
use core::mem::{self};
use core::pin::pin;
use core::task::{Context, Waker};

async fn test_async_drop<T>(x: T) {
    let mut x = mem::MaybeUninit::new(x);
    pin!(unsafe { async_drop_in_place(x.as_mut_ptr()) });
}

fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);

    let fut = pin!(async {
        test_async_drop(test_async_drop(0)).await;
    });
    fut.poll(&mut cx);
}
