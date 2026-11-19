//@ compile-flags: -Znext-solver
//@ check-pass
//@ edition: 2024

// An async function can hold a value across an await that contains an unsafe
// binder, but also something that needs drop. See #160270.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;

struct Adt<'a>(&'a u8);

async fn f(b: (unsafe<'a> MD<Adt<'a>>, Box<i32>)) {
    std::future::ready(()).await;
    drop(b);
}

fn main() {}
