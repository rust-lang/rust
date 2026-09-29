//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

// Pointer casts to and from binders where the metadata kind matches.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::ManuallyDrop as MD;

trait Tr {
    type Assoc<'a>: ?Sized;
}

fn slice_in(p: *const MD<[u8]>) -> *const unsafe<> MD<[u8]> {
    p as _
}
fn slice_out(p: *const unsafe<> MD<[u8]>) -> *const MD<[u8]> {
    p as _
}
fn dyn_in<'x>(p: *const MD<dyn Debug + 'x>) -> *const unsafe<'a> MD<dyn Debug + 'a> {
    p as _
}
fn to_thin(p: *const unsafe<'a> MD<dyn Debug + 'a>) -> *const u8 {
    p as _
}
fn param<U: ?Sized>(p: *const MD<U>) -> *const unsafe<'a> MD<(&'a u8, U)> {
    p as _
}
fn alias_identity<T: Tr>(
    p: *const unsafe<'a> MD<T::Assoc<'a>>,
) -> *const unsafe<'b> MD<T::Assoc<'b>> {
    p as _
}
fn alias_in<'x, T: Tr>(p: *const MD<T::Assoc<'x>>) -> *const unsafe<'a> MD<T::Assoc<'a>> {
    p as _
}

fn main() {}
