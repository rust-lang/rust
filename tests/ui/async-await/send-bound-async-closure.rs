//@ edition: 2021
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// This test verifies that we do not create a query cycle when typechecking has several inference
// variables that point to the same coroutine interior type.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

type ChannelTask = Pin<Box<dyn Future<Output = ()> + Send>>;

pub fn register_message_type() -> ChannelTask {
    Box::pin(async move {
        let f = |__cx: &mut Context<'_>| Poll::<()>::Pending;
        PollFn { f }.await
    })
}

struct PollFn<F> {
    f: F,
}

impl<F> Unpin for PollFn<F> {}

impl<T, F> Future for PollFn<F>
where
    F: FnMut(&mut Context<'_>) -> Poll<T>,
{
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        (&mut self.f)(cx)
    }
}

fn main() {}
