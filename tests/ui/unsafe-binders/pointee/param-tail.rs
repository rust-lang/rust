//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

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
