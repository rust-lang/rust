//@ check-pass
//@ edition:2018

#![feature(arbitrary_self_types)]

use std::task::{self, Poll};
use std::future::Future;
use std::marker::Unpin;
use std::pin::Pin;

// This is a regression test for an ICE/unbounded recursion issue relating to async-await.

#[derive(Debug)]
#[must_use = "futures do nothing unless polled"]
pub struct Lazy<F> {
    f: Option<F>
}

impl<F> Unpin for Lazy<F> {}

pub fn lazy<F, R>(f: F) -> Lazy<F>
    where F: FnOnce(&mut task::Context) -> R,
{
    Lazy { f: Some(f) }
}

impl<R, F> Future for Lazy<F>
    where F: FnOnce(&mut task::Context) -> R,
{
    type Output = R;

    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context) -> Poll<R> {
        Poll::Ready((self.f.take().unwrap())(cx))
    }
}

async fn __receive<WantFn, Fut>(want: WantFn) -> ()
    where Fut: Future<Output = ()>, WantFn: Fn(&Box<dyn Send + 'static>) -> Fut,
{
    lazy(|_| ()).await;
}

pub fn basic_spawn_receive() {
    async { __receive(|_| async { () }).await };
}

fn main() {}
