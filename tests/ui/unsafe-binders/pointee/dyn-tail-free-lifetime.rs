//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// If a `dyn Trait` does not mention any lifetimes from an unsafe binder, then
// the `DynMetadata` should be "normal" and not mention the binder or `ManuallyDrop`.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::ManuallyDrop as MD;
use std::ptr::{DynMetadata, Pointee};

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}

trait Tr<'x> {}

fn f<'x>() {
    metadata_is::<unsafe<'a> MD<(&'a u8, dyn Debug)>, DynMetadata<dyn Debug>>();
    metadata_is::<unsafe<'a> MD<dyn Tr<'x> + 'x>, DynMetadata<dyn Tr<'x> + 'x>>();
}

fn main() {}
