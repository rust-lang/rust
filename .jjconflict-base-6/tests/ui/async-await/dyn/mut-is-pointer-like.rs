//@ aux-build:block-on.rs
//@ edition: 2021
//@ known-bug: #133119

#![allow(refining_impl_trait)]
#![feature(async_fn_in_dyn_trait)]

extern crate block_on;

use std::future::Future;
use std::pin::Pin;

trait AsyncTrait {
    type Output;

    async fn async_dispatch(self: Pin<&mut Self>) -> Self::Output;
}

impl<F> AsyncTrait for F
where
    F: Future,
{
    type Output = F::Output;

    fn async_dispatch(self: Pin<&mut Self>) -> Pin<&mut Self> {
        self
    }
}

fn main() {
    block_on::block_on(async {
        let f = std::pin::pin!(async {
            println!("hello, world");
        });
        let x: Pin<&mut dyn AsyncTrait<Output = ()>> = f;
        x.async_dispatch().await;
    });
}
