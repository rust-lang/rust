//@ run-pass
//@ check-run-results
//@ revisions: with_feature without_feature
//@ aux-build:async-drop-dep.rs
//@ edition:2021

#![cfg_attr(with_feature, feature(async_drop))]
//[without_feature]~^ WARN found async drop types in dependecy `async_drop_dep`, but async_drop feature is disabled for `dependency_dropped`

#![allow(incomplete_features)]

extern crate async_drop_dep;

use async_drop_dep::MongoDrop;
use std::pin::pin;
use std::task::{Context, Poll, Waker};
use std::future::Future;

async fn asyncdrop() {
    let _ = MongoDrop::new().await;
}

pub fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

fn main() {
    let _ = block_on(asyncdrop());
}
