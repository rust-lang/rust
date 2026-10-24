//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// A generic unwrap_binder in a array length position should work (if otherwise allowed).

#![feature(unsafe_binders, generic_const_exprs)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

const fn mk<const N: usize>() -> unsafe<> usize {
    unsafe { wrap_binder!(N) }
}

fn f<const N: usize>() -> [u8; unsafe { unwrap_binder!(mk::<N>()) }] {
    todo!()
}

fn main() {}
