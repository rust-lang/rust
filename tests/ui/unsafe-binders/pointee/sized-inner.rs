//@ compile-flags: -Znext-solver
//@ check-pass

// Metadata of binders over sized types is `()`, and cannot mention the bound regions.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::ptr::{self, Pointee};

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}

struct BinderTail {
    b: unsafe<> (),
}

fn main() {
    metadata_is::<unsafe<'a> &'a u8, ()>();
    metadata_is::<unsafe<'a> (&'a u8, u32), ()>();
    metadata_is::<unsafe<> u8, ()>();
    metadata_is::<BinderTail, ()>();
    let _ = ptr::null::<BinderTail>();
}
