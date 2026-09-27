//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: unknown

// Nested unsafe binders should collapse into a single binder for metadata.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::ManuallyDrop as MD;
use std::ptr::{self, DynMetadata, Pointee};

trait Tr<'a> {}

trait Assoc {
    type Assoc<'a, 'b>: ?Sized;
}

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}
fn same_metadata<A: ?Sized, B: ?Sized>()
where
    A: Pointee<Metadata = <B as Pointee>::Metadata>,
{
}

// The bound regions of both binders only occur in sized types.
fn sized() {
    metadata_is::<unsafe<'a> MD<unsafe<'b> &'a &'b u8>, ()>();
    metadata_is::<unsafe<'a> MD<(&'a u8, unsafe<'b> &'b u8)>, ()>();
}

// The bound regions occur in the tail, but not in the metadata.
fn slice() {
    metadata_is::<unsafe<'a> MD<unsafe<'b> MD<[&'a &'b u8]>>, usize>();
}

// Only the inner binder's region occurs in the `dyn` tail.
type InnerOnly = unsafe<'a> MD<(&'a u8, unsafe<'b> MD<dyn Debug + 'b>)>;

// Only the outer binder's region occurs in the `dyn` tail.
type OuterOnly = unsafe<'a> MD<unsafe<'b> MD<(&'b u8, dyn Debug + 'a)>>;

// Both binders' regions occur in the `dyn` tail.
type Both = unsafe<'a> MD<unsafe<'b> MD<dyn Tr<'a> + 'b>>;

// As `Both`, with the outer region reaching the tail through a struct.
struct S<'x>(u8, unsafe<'b> MD<dyn Tr<'x> + 'b>);
type BothThroughStruct = unsafe<'a> MD<S<'a>>;

fn dyn_tail() {
    metadata_is::<InnerOnly, DynMetadata<unsafe<'b> MD<dyn Debug + 'b>>>();
    metadata_is::<OuterOnly, DynMetadata<unsafe<'a> MD<dyn Debug + 'a>>>();
    metadata_is::<Both, DynMetadata<unsafe<'a, 'b> MD<dyn Tr<'a> + 'b>>>();
    metadata_is::<BothThroughStruct, DynMetadata<unsafe<'a, 'b> MD<dyn Tr<'a> + 'b>>>();

    same_metadata::<InnerOnly, unsafe<'b> MD<dyn Debug + 'b>>();
    same_metadata::<OuterOnly, unsafe<'a> MD<dyn Debug + 'a>>();
    same_metadata::<Both, unsafe<'a, 'b> MD<dyn Tr<'a> + 'b>>();
    same_metadata::<Both, BothThroughStruct>();
}

// Metadata round-trips at the nested binder type.
fn round_trip(p: *const Both) -> *const Both {
    ptr::from_raw_parts(p as *const (), ptr::metadata(p))
}

// Generic alias tail mentioning both binders' regions. Without a where-clause
// the metadata is rigid, and a where-clause that holds for all regions of both
// binders fixes it.
fn alias_rigid<T: Assoc>(
    p: *const unsafe<'a> MD<unsafe<'b> MD<T::Assoc<'a, 'b>>>,
) -> *const unsafe<'a> MD<unsafe<'b> MD<T::Assoc<'a, 'b>>> {
    ptr::from_raw_parts(p as *const (), ptr::metadata(p))
}

fn alias_where_clause<T: Assoc>()
where
    for<'a, 'b> T::Assoc<'a, 'b>: Pointee<Metadata = usize>,
{
    metadata_is::<unsafe<'a> MD<unsafe<'b> MD<T::Assoc<'a, 'b>>>, usize>();
}

fn main() {}
