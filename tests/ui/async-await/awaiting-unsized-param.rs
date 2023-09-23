// edition: 2021
// revisions: no_drop_tracking drop_tracking_mir
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

#![feature(unsized_fn_params, unsized_locals)]
//~^ WARN the feature `unsized_locals` is incomplete

use std::future::Future;

async fn bug<T>(mut f: dyn Future<Output = T> + Unpin) -> T {
    //~^ ERROR the size for values of type `(dyn Future<Output = T> + Unpin + 'static)` cannot be known at compilation time
    (&mut f).await
}

fn main() {}
