//@ check-pass
//@ edition:2018

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::future::Future;

fn ergonomic_clone_async_closures() -> impl Future<Output = String> {
    let s = String::from("hi");

    async use {
        s
    }
}

fn main() {}
