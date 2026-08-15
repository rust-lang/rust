//@ edition: 2024
//@ compile-flags: -Znext-solver -Zdxf
//@ check-pass

use std::future::Future;
use std::task::{Context, Poll};
use std::pin::{Pin, pin};

#[derive(Clone)]
struct Foo;

pub enum MaybeDone<F: Future> {
    Future(F),
    Done(F::Output),
    Gone,
}

impl<F: Future<Output = ()>> Future for MaybeDone<F> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        Poll::Ready(())
    }
}

async fn do_work(_: Foo) {}

pub fn serve() -> impl Future<Output = ()> + Send {
    async move {
        let netstack = Foo;
        let work_fut = do_work(netstack.clone());
        let fut = pin!(MaybeDone::Future(work_fut));
        fut.await
    }
}

fn main() {}
