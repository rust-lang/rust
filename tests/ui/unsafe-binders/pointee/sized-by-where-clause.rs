//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// If a pointee ty is known to be `Sized`, then we know that metadata is `()`.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::ptr::{Pointee, Thin};

trait Tr {
    type Assoc<'a>;
}

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}
fn is_thin<T: ?Sized + Thin>() {}

fn f<T: Tr>()
where
    for<'a> T::Assoc<'a>: Copy,
{
    metadata_is::<unsafe<'a> T::Assoc<'a>, ()>();
    is_thin::<unsafe<'a> T::Assoc<'a>>();
    is_thin::<unsafe<'a> (u8, T::Assoc<'a>)>();
}

fn main() {}
