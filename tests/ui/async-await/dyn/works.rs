//@ aux-build:block-on.rs
//@ edition: 2021
//@ known-bug: #133119

#![allow(refining_impl_trait)]
#![feature(async_fn_in_dyn_trait)]

extern crate block_on;

use std::pin::Pin;
use std::future::Future;

trait AsyncTrait {
    async fn async_dispatch(&self);
}

impl AsyncTrait for &'static str {
    fn async_dispatch(&self) -> Pin<Box<impl Future<Output = ()>>> {
        Box::pin(async move {
            println!("message from the aether: {self}");
        })
    }
}

fn main() {
    block_on::block_on(async {
        let x: &dyn AsyncTrait = &"hello, world!";
        x.async_dispatch().await;
    });
}
