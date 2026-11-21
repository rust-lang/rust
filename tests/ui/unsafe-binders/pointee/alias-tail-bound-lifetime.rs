//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

// Tests where an alias ends up as the metadata ty and includes lifetimes
// from an unsafe binder.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::mem::ManuallyDrop as MD;
use std::ptr::{self, Pointee};

trait Tr {
    type Assoc<'a>: ?Sized;
}

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}
fn same_metadata<A: ?Sized, B: ?Sized>()
where
    A: Pointee<Metadata = <B as Pointee>::Metadata>,
{
}

struct W<'a, T: Tr>(u8, T::Assoc<'a>);

// This is rigid and we can't normalize this.
fn round_trip<T: Tr>(p: *const unsafe<'a> MD<T::Assoc<'a>>) -> *const unsafe<'a> MD<T::Assoc<'a>> {
    let m = ptr::metadata(p);
    let _copy = m;
    ptr::from_raw_parts(p as *const (), m)
}

// We should be able to use the where-clause to know that the metadata is `usize`.
fn higher_ranked_where_clause<T: Tr>()
where
    for<'a> T::Assoc<'a>: Pointee<Metadata = usize>,
{
    metadata_is::<unsafe<'a> MD<T::Assoc<'a>>, usize>();
}

// Same, for a `Sized` alias.
fn higher_ranked_sized<T: Tr>()
where
    for<'a> T::Assoc<'a>: Sized,
{
    metadata_is::<unsafe<'a> MD<T::Assoc<'a>>, ()>();
}

fn main() {}

// These are rigid and we can't normalize them.
fn nested_tail_rigid<T: Tr>() {
    same_metadata::<unsafe<'a> MD<(u8, T::Assoc<'a>)>, unsafe<'a> MD<T::Assoc<'a>>>();
    same_metadata::<unsafe<'a> MD<W<'a, T>>, unsafe<'a> MD<T::Assoc<'a>>>();
}

// Again, should be able to normalize from the where-clause.
fn nested_tail_where_clause<T: Tr>()
where
    for<'a> T::Assoc<'a>: Pointee<Metadata = usize>,
{
    metadata_is::<unsafe<'a> MD<(u8, T::Assoc<'a>)>, usize>();
    metadata_is::<unsafe<'a> MD<W<'a, T>>, usize>();
}

// And, similarly.
fn nested_tail_sized<T: Tr>()
where
    for<'a> T::Assoc<'a>: Sized,
{
    metadata_is::<unsafe<'a> MD<(u8, T::Assoc<'a>)>, ()>();
    metadata_is::<unsafe<'a> MD<W<'a, T>>, ()>();
}
