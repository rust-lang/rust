//@ known-bug: #140531
//@compile-flags: -Zlint-mir --crate-type lib
//@ edition:2024
#![feature(async_drop)]
async fn call_once(f: impl AsyncFnOnce()) {
    let fut = Box::pin(f());
}
