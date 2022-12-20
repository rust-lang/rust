// check-pass
// compile-flags: --edition 2018

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

async fn new_future() {
    loop {}
}

pub struct SelectAll<Fut> {
    _inner: Fut,
}

fn select_all<I>(_: I) -> SelectAll<I::Item>
where
    I: IntoIterator,
    I::Item: Future + Unpin,
{
    loop {}
}

impl<Fut: Future> Future for SelectAll<Fut> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        loop {}
    }
}

async fn run_one_step() {
    let mut select = select_all(vec![Box::pin(new_future())]);
    let poll_fns = &mut |cx: &mut Context<'_>| Pin::new(&mut select).poll(cx);
    for _poll_fn in [poll_fns] {}
}

fn main() {}
