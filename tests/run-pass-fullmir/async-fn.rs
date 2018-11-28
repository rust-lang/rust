// FIXME: investigate why this fails since barriers have been added
// compile-flags: -Zmiri-disable-validation

#![feature(
    async_await,
    await_macro,
    futures_api,
    pin,
)]

use std::{future::Future, pin::Pin, task::Poll};

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

fn main() {
    use std::{sync::Arc, task::{Wake, local_waker}};

    struct NoWake;
    impl Wake for NoWake {
        fn wake(_arc_self: &Arc<Self>) {
            panic!();
        }
    }

    let lw = unsafe { local_waker(Arc::new(NoWake)) };
    let x = 5;
    let mut fut = foo(&x, 7);
    assert_eq!(unsafe { Pin::new_unchecked(&mut fut) }.poll(&lw), Poll::Ready(31));
}
