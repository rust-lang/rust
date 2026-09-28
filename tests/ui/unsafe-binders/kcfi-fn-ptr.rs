//@ compile-flags: -Znext-solver
//@ build-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""
//@ needs-sanitizer-kcfi
//@ no-prefer-dynamic
//@ compile-flags: -Cpanic=abort -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ compile-flags: -Zunstable-options -Csymbol-mangling-version=legacy
//@ ignore-backends: gcc

// KCFI sanitizer works with unsafe binders.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::wrap_binder;

pub fn take(_: unsafe<'a> &'a u8) {}

fn main() {
    let f: fn(unsafe<'a> &'a u8) = take;
    f(unsafe { wrap_binder!(&0) });
}
