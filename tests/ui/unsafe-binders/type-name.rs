//@ compile-flags: -Znext-solver
//@ build-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// `type_name` includes the unsafe binder.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::any::type_name;

fn main() {
    assert_eq!(type_name::<unsafe<'a> &'a ()>(), "unsafe<> &'_ ()");
    assert_eq!(type_name::<unsafe<'a> (&'a u8, u8)>(), "unsafe<> (&'_ u8, u8)");
}
