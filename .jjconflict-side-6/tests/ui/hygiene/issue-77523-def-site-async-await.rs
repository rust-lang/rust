//@ build-pass
//@ aux-build:def-site-async-await.rs
//@ ignore-backends: gcc

// Regression test for issue #77523
// Tests that we don't ICE when an unusual combination
// of def-site hygiene and cross-crate monomorphization occurs.

extern crate def_site_async_await;

use std::future::Future;

fn mk_ctxt() -> std::task::Context<'static> {
    panic!()
}

fn main() {
    Box::pin(def_site_async_await::serve()).as_mut().poll(&mut mk_ctxt());
}
