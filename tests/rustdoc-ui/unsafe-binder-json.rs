//@ compile-flags: -Znext-solver -Zunstable-options --output-format=json
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "((compiler|src)/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// rustdoc's JSON output should be able to represent unsafe binders

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

pub fn f(_: unsafe<'a> &'a u8) {}
