//@ compile-flags: -Znext-solver
//@ check-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// The metadata of a type with a `dyn Trait` under an unsafe binder is a bit weird:
// It ends up being a `DynMetadata<unsafe<'a> ManuallyDrop<dyn Trait + 'a>>>`,
// because this is the only way to express the metadata in such a way that is
// well-formed, meets the `Pointee::Metadata` bounds, and doesn't transmute
// unsafe binder lifetimes to `'static`.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::ManuallyDrop as MD;
use std::ptr::{self, DynMetadata, Pointee};

trait Tr<'a> {}

type B1 = unsafe<'a> MD<dyn Debug + 'a>;
type B2 = unsafe<'a> MD<dyn Tr<'a> + 'a>;
type B3 = unsafe<'a> MD<dyn Iterator<Item = &'a u8> + 'a>;
type B4 = unsafe<'a> MD<(&'a u8, dyn Debug + 'a)>;

// The `dyn` tail is reached through a struct that isn't `ManuallyDrop`. Only the
// type directly inside the binder has to be `Copy` or `ManuallyDrop`, so `W` can
// be unsized.
struct W<'a>(u8, MD<dyn Debug + 'a>);
type B5 = unsafe<'a> MD<(u8, W<'a>)>;

// Only the bound regions that occur in the `dyn` tail are re-bound.
type B6 = unsafe<'a, 'b> MD<(&'b u8, dyn Tr<'a> + 'a)>;

// Nested `ManuallyDrop`s collapse.
type B7 = unsafe<'a> MD<MD<dyn Debug + 'a>>;

fn metadata_is<T: ?Sized + Pointee<Metadata = M>, M>() {}

fn exact_metadata() {
    metadata_is::<B1, DynMetadata<unsafe<'a> MD<dyn Debug + 'a>>>();
    metadata_is::<B2, DynMetadata<unsafe<'a> MD<dyn Tr<'a> + 'a>>>();
    metadata_is::<B3, DynMetadata<unsafe<'a> MD<dyn Iterator<Item = &'a u8> + 'a>>>();
    metadata_is::<B4, DynMetadata<unsafe<'a> MD<dyn Debug + 'a>>>();
    metadata_is::<B5, DynMetadata<unsafe<'a> MD<dyn Debug + 'a>>>();
    metadata_is::<B6, DynMetadata<unsafe<'a> MD<dyn Tr<'a> + 'a>>>();
    metadata_is::<B7, DynMetadata<unsafe<'a> MD<dyn Debug + 'a>>>();
}

// Free regions in the `dyn` stay free; only the bound ones are re-bound.
fn mixed_free_and_bound<'x>() {
    metadata_is::<unsafe<'a> MD<dyn Tr<'x> + 'a>, DynMetadata<unsafe<'a> MD<dyn Tr<'x> + 'a>>>();
}

fn round_trip_b1(p: *const B1) -> *const B1 {
    ptr::from_raw_parts(p as *const (), ptr::metadata(p))
}
fn round_trip_b2(p: *const B2) -> *const B2 {
    ptr::from_raw_parts(p as *const (), ptr::metadata(p))
}
fn round_trip_b3(p: *const B3) -> *const B3 {
    ptr::from_raw_parts(p as *const (), ptr::metadata(p))
}
fn round_trip_b4(p: *const B4) -> *const B4 {
    ptr::from_raw_parts(p as *const (), ptr::metadata(p))
}
fn round_trip_b5(p: *const B5) -> *const B5 {
    ptr::from_raw_parts(p as *const (), ptr::metadata(p))
}

// `Pointee::Metadata`'s item bounds must hold for whatever the metadata is.
fn item_bounds(p: *const B1) {
    fn bounds<M: Copy + Send + Sync + Ord + std::hash::Hash + Unpin + Debug>(_: M) {}
    bounds(ptr::metadata(p));
}

// The metadata is a `DynMetadata`, so its methods are available.
fn dyn_metadata_methods(p: *const B1) -> usize {
    ptr::metadata(p).size_of()
}

fn main() {}
