//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""
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
