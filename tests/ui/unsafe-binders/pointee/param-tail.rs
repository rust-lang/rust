//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// If the metadata ty under an unsafe binder doesn't mention any bound lifetimes,
// then it should be "normal" and not mention the binder or `ManuallyDrop`.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;
use std::ptr::Pointee;

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}
fn same_metadata<A: ?Sized, B: ?Sized>()
where
    A: Pointee<Metadata = <B as Pointee>::Metadata>,
{
}

fn unsized_param<T: ?Sized>() {
    same_metadata::<unsafe<'a> MD<T>, T>();
    same_metadata::<unsafe<'a> MD<(&'a u8, T)>, T>();
}

fn sized_param<T: Copy>() {
    metadata_is::<unsafe<'a> T, ()>();
    metadata_is::<unsafe<'a> (&'a u8, T), ()>();
}

fn main() {}
