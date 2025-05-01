//! This test used to ICE: rust-lang/rust#128695
//! Fixed when re-work async drop to shim drop glue coroutine scheme.
//@ edition: 2021

use core::pin::{pin, Pin};

fn main() {
    let fut = pin!(async {
        let async_drop_fut = pin!(core::future::async_drop(async {})); //~ ERROR: expected function, found module `core::future::async_drop`
        //~^ ERROR: module `async_drop` is private
        (async_drop_fut).await;
    });
}
