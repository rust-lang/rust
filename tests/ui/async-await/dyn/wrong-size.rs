//@ edition: 2021
//@ known-bug: #133119

#![feature(async_fn_in_dyn_trait)]

use std::future::Future;

trait AsyncTrait {
    async fn async_dispatch(&self);
}

impl AsyncTrait for &'static str {
    fn async_dispatch(&self) -> impl Future<Output = ()> {
        async move {
            // The implementor must box the future...
        }
    }
}

fn main() {
    let x: &dyn AsyncTrait = &"hello, world!";
    // FIXME ~^ ERROR `impl Future<Output = ()>` needs to have the same ABI as a pointer
}
