//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

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
