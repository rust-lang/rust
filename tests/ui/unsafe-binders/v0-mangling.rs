//@ compile-flags: -Znext-solver
//@ build-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""
//@ compile-flags: -Csymbol-mangling-version=v0

// v0 symbol mangling works with unsafe binders. See #154367.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

fn generic<T>() {}

fn main() {
    generic::<unsafe<'a> &'a ()>();
}
