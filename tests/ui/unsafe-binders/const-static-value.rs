//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// Constants and statics can hold unsafe binders, and const-eval validates the
// value inside a binder like any other value. See #153362.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::wrap_binder;

const C: unsafe<> u8 = unsafe { wrap_binder!(1u8) };
static S: unsafe<'a> &'a u8 = unsafe { wrap_binder!(&1u8) };

fn main() {
    let _x = C;
}
