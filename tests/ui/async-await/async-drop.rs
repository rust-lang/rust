//@ run-pass
//@ check-run-results

#![feature(async_drop, impl_trait_in_assoc_type, noop_waker)]
#![allow(incomplete_features)]

//@ edition: 2021

use core::future::{async_drop, AsyncDrop, Future};
use core::hint::black_box;
use core::mem::ManuallyDrop;
use core::pin::{pin, Pin};
use core::task::{Context, Poll, Waker};

fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);

    let i = 13;
    // TODO: Check idempotency
    let fut = pin!(async {
        async_drop(Bar(0)).await;
        async_drop(Foo(0)).await;
        async_drop([Foo(1), Foo(2)]).await;
        async_drop((Foo(3), Foo(4))).await;
        async_drop(5).await;
        let j = 42;
        async_drop(&i).await;
        async_drop(&j).await;
        async_drop(Baz { _b: Foo(8), _a: Foo(7), n: 6 }).await;
        async_drop(ManuallyDrop::new(Foo(9))).await;

        let foo = Foo(10);
        async_drop(Qux { foo: &foo }).await;

        let foo = Foo(11);
        async_drop(|| black_box(foo)).await;

        let foo = Foo(13);
        async_drop(Fiz::A(Foo(12))).await;
        async_drop(Fiz::B(Qux { foo: &foo })).await;
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

struct Qux<'a> {
    foo: &'a Foo,
}

impl AsyncDrop for Qux<'_> {
    type Dropper<'b> = impl Future<Output = ()> + 'b
        where Self: 'b;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!("<Qux as AsyncDrop>::Dropper::poll: {}", self.foo.0);
        }
    }
}

#[allow(dead_code)]
struct Bar(i32);

struct Baz{
    _a: Foo,
    _b: Foo,
    n: i32,
}

impl AsyncDrop for Baz {
    type Dropper<'a> = impl Future<Output = ()> + 'a;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            println!("<Baz as AsyncDrop>::Dropper::poll: {}", self.n);
        }
    }
}

#[allow(dead_code)]
enum Fiz<'a> {
    A(Foo),
    B(Qux<'a>),
}
