//@ check-pass
//@ edition:2018

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::future::Future;
use std::sync::Arc;

fn ergonomic_clone_async_closures() -> impl Future<Output = Arc<String>> {
    let s = Arc::new(String::from("hi"));

    async { s.use }
}

fn main() {}
