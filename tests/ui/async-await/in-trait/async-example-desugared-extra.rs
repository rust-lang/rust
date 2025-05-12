//@ check-pass
//@ edition: 2021

use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

pub trait MyTrait {
    #[allow(async_fn_in_trait)]
    async fn foo(&self) -> i32;
}

#[derive(Clone)]
struct MyFuture(i32);

impl Future for MyFuture {
    type Output = i32;
    fn poll(
        self: Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> Poll<<Self as Future>::Output> {
        Poll::Ready(self.0)
    }
}

impl MyTrait for i32 {
    #[expect(refining_impl_trait)]
    fn foo(&self) -> impl Future<Output = i32> + Clone {
        MyFuture(*self)
    }
}

fn main() {}
