//@revisions: stack tree
//@compile-flags: -Zmiri-strict-provenance
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(async_drop, impl_trait_in_assoc_type, noop_waker)]
#![allow(incomplete_features)]

use core::future::{async_drop, async_drop_in_place, AsyncDrop, Future};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::pin::{pin, Pin};
use core::task::{Context, Poll, Waker};

fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);

    let i = 13;
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

        // This is a temporary test for fused futures before enums get
        // support too
        let mut fiz = MaybeUninit::new(Fiz::A(10));
        let mut fut = pin!(unsafe { async_drop_in_place(fiz.as_mut_ptr()) });
        fut.as_mut().await;
        fut.await;

        let foo = Foo(11);
        async_drop(Qux { foo: &foo }).await;
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

enum Fiz {
    A(i32),
}

impl AsyncDrop for Fiz {
    type Dropper<'a> = impl Future<Output = ()> + 'a;

    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_> {
        async move {
            match &*self {
                Fiz::A(i) => println!("<Fiz::A as AsyncDrop>::Dropper::poll: {i}"),
            }
        }
    }
}

#[allow(dead_code)]
struct Bar(i32);

struct Baz {
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
