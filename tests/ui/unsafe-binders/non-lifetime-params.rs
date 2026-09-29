//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// Unsafe binders cannot bind type and const params. See #141293.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

type WithType = unsafe<T> ();
type WithConst = unsafe<const N: usize> [u8; N];

fn main() {}
