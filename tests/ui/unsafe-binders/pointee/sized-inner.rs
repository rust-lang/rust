//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// Metadata of binders over sized types is `()`, and cannot mention the bound regions.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::ptr::{self, Pointee};

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}

struct BinderTail {
    b: unsafe<> (),
}

fn main() {
    metadata_is::<unsafe<'a> &'a u8, ()>();
    metadata_is::<unsafe<'a> (&'a u8, u32), ()>();
    metadata_is::<unsafe<> u8, ()>();
    metadata_is::<BinderTail, ()>();
    let _ = ptr::null::<BinderTail>();
}
