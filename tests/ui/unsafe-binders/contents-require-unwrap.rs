//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// It should not be possible to read the discriminant or otherwise destructure
// and unsafe binder with unwrapping. See #158839.

#![feature(unsafe_binders, variant_count)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

fn destructure(b: unsafe<'a> (&'a u8, u8)) {
    let (_x, _y) = b;
}

fn match_variant(b: unsafe<> Option<u8>) {
    match b {
        Some(_) => {}
        None => {}
    }
}

fn if_let(b: unsafe<> Option<u8>) {
    if let Some(_) = b {}
}

fn wildcard(b: unsafe<> u8) {
    match b {
        _ => {}
    }
    let _ = b;
}

fn variant_count() {
    let _ = std::mem::variant_count::<unsafe<> Option<u8>>();
}

fn unwrapped(b: unsafe<'a> (&'a u8, Option<u8>)) -> u8 {
    unsafe {
        let (x, y) = unwrap_binder!(b);
        let _ = std::mem::discriminant(&y);
        if let Some(y) = y { *x + y } else { *x }
    }
}

fn main() {
    let b: unsafe<> Option<u8> = unsafe { wrap_binder!(Some(1)) };
    let _ = std::mem::discriminant(&b);
}
