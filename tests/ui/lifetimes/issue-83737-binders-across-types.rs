//@ build-pass
//@ edition: 2018
//@ compile-flags: --crate-type rlib

use std::future::Future;

async fn handle<F>(slf: &F)
where
    F: Fn(&()) -> Box<dyn Future<Output = ()> + Unpin>,
{
    (slf)(&()).await;
}

fn main() {}
