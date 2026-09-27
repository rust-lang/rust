//@ compile-flags: -Znext-solver
//@ check-pass

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
