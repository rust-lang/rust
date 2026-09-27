//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: unknown

// Any lifetimes within the type ascriptions on `wrap_binder!` or `unwrap_binder!`
// should be enforced by borrowck.
//
// For now, we just error on this - because this has not been implemented yet.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::cell::Cell;
use std::unsafe_binder::{unwrap_binder, wrap_binder};

// The annotation drives inference: without it, both of these are ambiguous.
fn wrap_infers() {
    let b = unsafe { wrap_binder!(&42; unsafe<'a> &'a i32) };
    let _x: i32 = unsafe { *unwrap_binder!(b) };
}
fn unwrap_infers() {
    let f = |b| unsafe { *unwrap_binder!(b; unsafe<'a> &'a i32) };
    let _x: i32 = f(unsafe { wrap_binder!(&0) });
}

// The annotation must agree with the expected type, with the wrapped value,
// and with the unwrapped operand.
// error
fn wrap_mismatch_expected() {
    let _x: unsafe<'a> &'a i32 = unsafe { wrap_binder!(&0; unsafe<'a> &'a u32) };
}
// error
fn wrap_mismatch_bound_vars() {
    let _x: unsafe<'a> &'a u8 = unsafe { wrap_binder!(&0; unsafe<'a, 'b> &'a &'b u8) };
}
// error
fn wrap_mismatch_value() {
    let _x = unsafe { wrap_binder!(&0i64; unsafe<'a> &'a u32) };
}
// error
fn unwrap_mismatch(b: unsafe<'a> &'a i32) {
    let _x = unsafe { unwrap_binder!(b; unsafe<'a> &'a u32) };
}

// The annotation must be an unsafe binder.
// error
fn wrap_not_binder() {
    let _x = unsafe { wrap_binder!(0; i32) };
}
// error
fn unwrap_not_binder() {
    let _x = unsafe { unwrap_binder!(0; i32) };
}

// The unwrap operand only has to be a subtype of the annotation, so a more
// general binder is accepted.
fn unwrap_more_general(b: unsafe<'a> &'a u8, c: unsafe<'a> (&'a u8, &'static u8)) {
    let _x = unsafe { unwrap_binder!(b; unsafe<> &'static u8) };
    let _x = unsafe { unwrap_binder!(c; unsafe<'a> (&'a u8, &'a u8)) };
}

fn wrap_empty<'x>(r: &'x u8) {
    let _x = unsafe { wrap_binder!(r; unsafe<> &'static u8) };
}
fn wrap_free<'x>(r: &'x u8) {
    let _x = unsafe { wrap_binder!((&0, r); unsafe<'a> (&'a u8, &'static u8)) };
}

fn wrap_expected_free<'x, 'y>(r: &'x u8) -> unsafe<> &'x u8 {
    unsafe { wrap_binder!(r; unsafe<> &'y u8) }
}
fn wrap_expected_bound(c: &'static Cell<&'static u8>) {
    let _x: unsafe<'a> &'a Cell<&'static u8> =
        unsafe { wrap_binder!(c; unsafe<'a> &'a Cell<&'a u8>) };
}

fn unwrap_value<'x>(b: unsafe<'a> (&'a u8, &'x u8)) {
    let _x = unsafe { unwrap_binder!(b; unsafe<'a> (&'a u8, &'static u8)) };
}
fn unwrap_place<'x>(p: *const unsafe<'a> (&'a u8, &'x u8)) {
    let _x = unsafe { &raw const unwrap_binder!(*p; unsafe<'a> (&'a u8, &'static u8)) };
}

fn main() {}
