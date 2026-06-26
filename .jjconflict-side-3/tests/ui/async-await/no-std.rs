//@ edition:2018
//@ check-pass

#![no_std]
#![crate_type = "rlib"]

use core::future::Future;

async fn a(f: impl Future) {
    f.await;
}

fn main() {}
