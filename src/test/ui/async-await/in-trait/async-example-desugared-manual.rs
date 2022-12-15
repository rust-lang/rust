// edition: 2021

#![feature(async_fn_in_trait)]
#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;
use std::task::Poll;

trait MyTrait {
    async fn foo(&self) -> i32;
}

struct MyFuture;
impl Future for MyFuture {
    type Output = i32;
    fn poll(self: std::pin::Pin<&mut Self>, _: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(0)
    }
}

impl MyTrait for u32 {
    fn foo(&self) -> MyFuture {
        //~^ ERROR method `foo` should be async
        MyFuture
    }
}

fn main() {}
