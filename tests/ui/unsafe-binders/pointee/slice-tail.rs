//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// Binders over unsized types whose metadata is `usize`.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;
use std::ptr::Pointee;

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}

struct S(u8, unsafe<'a> MD<[&'a u8]>);

// The bound region only appears in a sized field.
struct W<'a, T: ?Sized>(&'a u8, T);

fn main() {
    metadata_is::<unsafe<> MD<[u8]>, usize>();
    metadata_is::<unsafe<'a> MD<[&'a u8]>, usize>();
    metadata_is::<unsafe<'a> MD<(&'a u8, str)>, usize>();
    metadata_is::<unsafe<'a> MD<W<'a, [u8]>>, usize>();
    metadata_is::<S, usize>();
}
