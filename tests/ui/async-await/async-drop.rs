//@ run-pass
//@ check-run-results

#![feature(async_drop, impl_trait_in_assoc_type, noop_waker)]
#![allow(incomplete_features)]

//@ edition: 2021

use core::future::{async_drop, AsyncDrop, Future};
use core::pin::{pin, Pin};
use core::task::{Context, Poll, Waker};


fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);

    let fut = pin!(async {
        async_drop(Bar(0)).await;
        async_drop(Foo(0)).await;
        async_drop([Foo(1), Foo(2)]).await;
    });

    let res = fut.poll(&mut cx);
    assert_eq!(res, Poll::Ready(()));
}


struct Foo(i32);

impl AsyncDrop for Foo {
    type Dropper<'a> = impl Future<Output = ()> + 'a;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!("<Foo as AsyncDrop>::Dropper::poll: {}", self.0);
        }
    }
}

#[allow(dead_code)]
struct Bar(i32);
