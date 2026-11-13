//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

// For an opaque type, we duplicate captured lifetimes (the parent args and the
// opaque's own args). Although the parent args are marked as *bivariant*, in an
// invariant context, those are still considered invariant and participate in
// outlives relationships. Therefore, we must mark them as live.
//
// This specific test demonostrates the path where there are no captured regions.

#![forbid(unsafe_code)]

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

trait Id {
    type Out: ?Sized;
}
impl<X: ?Sized> Id for X {
    type Out = X;
}

fn f<T: std::fmt::Display>(t: <&T as Id>::Out) -> impl std::fmt::Display {
    t
}

fn through_closure() {
    let mut x = 0;
    let a = (|| f(&x))();
    x = 1;
    //~^ ERROR cannot assign to `x` because it is borrowed
    println!("{a}");
}

async fn multiple_named_lifetimes<'a, 'b>(a: <&'a u8 as Id>::Out, _: &u8) -> &'a u8 {
    a
}

fn through_closure_async() {
    let mut a = 22u8;
    let b = 44u8;
    let future = (|| multiple_named_lifetimes(&a, &b))();
    a += 1;
    //~^ ERROR cannot assign to `a` because it is borrowed
    let mut fut = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    if let Poll::Ready(r) = fut.as_mut().poll(&mut cx) {
        println!("{}", *r);
    }
}

fn main() {
    through_closure();
    through_closure_async();
}
