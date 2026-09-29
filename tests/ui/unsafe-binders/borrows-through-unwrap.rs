//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// We should be able to borrowck through place projections through an `unwrap_binder!()`.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::unwrap_binder;

fn two_borrows(mut b: unsafe<> (u8, u8)) {
    unsafe {
        let x = &mut unwrap_binder!(b).0;
        let y = &unwrap_binder!(b).1;
        *x = *y;
    }
}

fn guarded_match(b: unsafe<> Option<u8>) -> u8 {
    unsafe {
        match unwrap_binder!(b) {
            Some(x) if x > 0 => x,
            _ => 0,
        }
    }
}

fn main() {}
