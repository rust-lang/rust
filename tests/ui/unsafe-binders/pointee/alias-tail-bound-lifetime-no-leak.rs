//@ compile-flags: -Znext-solver

// We should only be able to normalize an alias within an unsafe binder if that
// normalization would hold when universally qualified over the bound regions.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;
use std::ptr::Pointee;

trait Tr {
    type Assoc<'a>: ?Sized;
}

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}

fn only_static<T: Tr>()
where
    T::Assoc<'static>: Pointee<Metadata = usize>,
{
    metadata_is::<unsafe<'a> MD<T::Assoc<'a>>, usize>(); //~ ERROR type mismatch resolving
}

fn only_one<'x, T: Tr>()
where
    T::Assoc<'x>: Pointee<Metadata = usize>,
{
    metadata_is::<unsafe<'a> MD<T::Assoc<'a>>, usize>(); //~ ERROR type mismatch resolving
}

fn main() {}
