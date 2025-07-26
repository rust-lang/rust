// ex-ice: #140531
//@ compile-flags: -Zlint-mir --crate-type lib
//@ edition:2024
//@ check-pass

#![feature(async_drop)]
#![allow(incomplete_features)]

async fn call_once(f: impl AsyncFnOnce()) {
    let fut = Box::pin(f());
}
