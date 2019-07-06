// This used to compile the future down to ud2, due to uninhabited types being
// handled incorrectly in generators.
// compile-flags: -Copt-level=z -Cdebuginfo=2 --edition=2018

#![feature(async_await)]
#![allow(unused)]

use std::future::Future;
use std::task::Poll;
use std::task::Context;
use std::pin::Pin;
use std::rc::Rc;

struct Never();
impl Future for Never {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

fn main() {
    let fut = async {
        let _rc = Rc::new(()); // Also crashes with Arc
        Never().await;
    };
    let _bla = fut; // Moving the future is required.
}
