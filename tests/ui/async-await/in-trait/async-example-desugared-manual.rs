// edition: 2021

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
