// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2018
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

fn spawn<T: Send>(_: T) {}

pub struct AFuture;
impl Future for AFuture{
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        unimplemented!()
    }
}

async fn foo() {
    spawn(async { //~ ERROR future cannot be sent between threads safely
        let a = std::ptr::null_mut::<()>(); // `*mut ()` is not `Send`
        AFuture.await;
        drop(a);
    });
}

fn main() {}
