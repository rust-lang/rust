//@ edition: 2021

#![feature(unsized_fn_params)]

use std::future::Future;

async fn bug<T>(mut f: dyn Future<Output = T> + Unpin) -> T {
    //~^ ERROR the size for values of type `dyn Future<Output = T> + Unpin` cannot be known at compilation time
    (&mut f).await
}

fn main() {}
