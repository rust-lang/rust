//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""
//@ compile-flags: -Zunpretty=stable-mir

// `-Zunpretty=stable-mir` prints MIR that wraps and unwraps unsafe binders.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

pub fn f(x: u8) -> u8 {
    let b: unsafe<> u8 = unsafe { wrap_binder!(x) };
    unsafe { unwrap_binder!(b) }
}

fn main() {}
