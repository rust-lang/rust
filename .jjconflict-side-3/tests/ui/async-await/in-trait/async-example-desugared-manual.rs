//@ edition: 2021
//@ check-pass

use std::future::Future;
use std::task::Poll;

#[allow(async_fn_in_trait)]
pub trait MyTrait {
    async fn foo(&self) -> i32;
}

pub struct MyFuture;
impl Future for MyFuture {
    type Output = i32;
    fn poll(self: std::pin::Pin<&mut Self>, _: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(0)
    }
}

impl MyTrait for u32 {
    #[warn(refining_impl_trait)]
    fn foo(&self) -> MyFuture {
        //~^ WARN impl trait in impl method signature does not match trait method signature
        MyFuture
    }
}

fn main() {}
