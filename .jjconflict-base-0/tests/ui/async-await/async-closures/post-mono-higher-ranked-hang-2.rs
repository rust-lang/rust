//@ edition: 2021
//@ build-fail

// Regression test for <https://github.com/rust-lang/rust/issues/135780>.

use std::future::Future;
use std::ops::AsyncFn;
use std::pin::Pin;

fn recur<'l>(closure: &'l impl AsyncFn()) -> Pin<Box<dyn Future<Output = ()> + 'l>> {
    Box::pin(async move {
        let _ = closure();
        let _ = recur(&async || {
            //~^ ERROR reached the recursion limit
            let _ = closure();
        });
    })
}

fn main() {
    let closure = async || {};
    let _ = recur(&closure);
}
