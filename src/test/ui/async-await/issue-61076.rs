// edition:2018

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

struct T;

impl Future for T {
    type Output = Result<(), ()>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

async fn foo() -> Result<(), ()> {
    Ok(())
}

async fn bar() -> Result<(), ()> {
    foo()?; //~ ERROR the `?` operator can only be applied to values that implement `std::ops::Try`
    Ok(())
}

async fn baz() -> Result<(), ()> {
    let t = T;
    t?; //~ ERROR the `?` operator can only be applied to values that implement `std::ops::Try`
    Ok(())
}

fn main() {}
