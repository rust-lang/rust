// run-pass
// edition:2018

#![allow(unused)]

use core::future::Future;
use core::pin::Pin;
use core::task::{Poll, Context};

struct Map<Fut, F>(Fut, F);

impl<Fut, F, S> Future for Map<Fut, F> where Fut: Future, F: FnOnce(Fut::Output) -> S {
    type Output = S;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}

enum MaybeDone<Fut: Future> {
    Maybe(Fut),
    Done(Fut::Output),
}

impl<Fut: Future> Future for MaybeDone<Fut> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}

fn drop_static(s: &'static str) {}

async fn foo() {
    let a = async { "hello" as &'static str};
        let b = Map(a, drop_static);
        let c = MaybeDone::Maybe(b);
        c.await
}

fn needs_send<T: Send>(t: T) {}

fn main() {
    needs_send(foo());
}
