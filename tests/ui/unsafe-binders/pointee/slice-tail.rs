//@ compile-flags: -Znext-solver
//@ check-pass

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
