//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

// Casting out of an unsafe binder should require an `unwrap_binder!` call..

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::fmt::Debug;
use std::mem::ManuallyDrop as MD;
use std::unsafe_binder::unwrap_binder;

trait Tr<'a> {}

fn direct<'x>(p: *const MD<dyn Debug + 'x>) -> *const MD<dyn Debug + 'static> {
    p as _
}

fn through_binder<'x>(p: *const MD<dyn Debug + 'x>) -> *const MD<dyn Debug + 'static> {
    let q = p as *const unsafe<'a> MD<dyn Debug + 'a>;
    q as _
}

// Casting out can't pick any lifetime for the bound region, not just `'static`.
fn out_to_param<'x>(q: *const unsafe<'a> MD<dyn Debug + 'a>) -> *const MD<dyn Debug + 'x> {
    q as _
}

// The same holds for bound regions in the principal trait's arguments, for
// trait objects without a principal, and for `dyn` tails reached through a tuple.
fn out_principal_arg(
    q: *const unsafe<'a> MD<dyn Tr<'a> + 'static>,
) -> *const MD<dyn Tr<'static> + 'static> {
    q as _
}
fn out_no_principal(q: *const unsafe<'a> MD<dyn Send + 'a>) -> *const MD<dyn Send + 'static> {
    q as _
}
fn out_through_tuple(
    q: *const unsafe<'a> MD<(u8, dyn Debug + 'a)>,
) -> *const MD<(u8, dyn Debug + 'static)> {
    q as _
}

// The same holds for `*mut` pointers.
fn out_mut(q: *mut unsafe<'a> MD<dyn Debug + 'a>) -> *mut MD<dyn Debug + 'static> {
    q as _
}
fn out_const_from_mut(q: *mut unsafe<'a> MD<dyn Debug + 'a>) -> *const MD<dyn Debug + 'static> {
    q as _
}

// Casting into a binder, and between binders, is fine: the target's bound
// regions can be anything.
fn into_binder<'x>(p: *const MD<dyn Tr<'x> + 'x>) -> *const unsafe<'a> MD<dyn Tr<'a> + 'a> {
    p as _
}
fn binder_to_binder(
    q: *const unsafe<'a> MD<dyn Tr<'a> + 'a>,
) -> *const unsafe<'b> MD<dyn Tr<'b> + 'b> {
    q as _
}

// Unwrapping through the pointer is the way out.
fn out_via_unwrap(q: *const unsafe<'a> MD<dyn Debug + 'a>) -> *const MD<dyn Debug + 'static> {
    unsafe { &raw const unwrap_binder!(*q) }
}

fn into_binder_mut<'x>(p: *mut MD<dyn Debug + 'x>) -> *mut unsafe<'a> MD<dyn Debug + 'a> {
    p as _
}
fn out_via_unwrap_mut(q: *mut unsafe<'a> MD<dyn Debug + 'a>) -> *mut MD<dyn Debug + 'static> {
    unsafe { &raw mut unwrap_binder!(*q) }
}

fn main() {}
