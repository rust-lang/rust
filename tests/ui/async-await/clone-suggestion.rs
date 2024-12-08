//@ edition: 2021
//@ run-rustfix

#![allow(unused)]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

#[derive(Clone)]
struct SharedFuture;

impl Future for SharedFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<<Self as Future>::Output> {
        todo!()
    }
}

async fn foo() {
    let f = SharedFuture;
    f.await;
    f.await;
    //~^ ERROR use of moved value
}

fn main() {}
