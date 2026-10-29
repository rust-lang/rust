//@ compile-flags: -Znext-solver

// The operand of `wrap_binder!` isn't a coercion site. The result of
// `unwrap_binder!` coerces like any other expression, but its operand
// isn't auto-dereferenced.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::unsafe_binder::{unwrap_binder, wrap_binder};

fn reborrow(x: &mut u8) -> unsafe<'a> &'a u8 {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn unsize(x: &[u8; 2]) -> unsafe<'a> &'a [u8] {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn unsize_dyn(x: &u8) -> unsafe<'a> &'a (dyn Debug + 'a) {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn deref(x: &String) -> unsafe<'a> &'a str {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn deref_box(x: &Box<u8>) -> unsafe<'a> &'a u8 {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn ref_to_ptr(x: &u8) -> unsafe<> *const u8 {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn mut_to_ptr(x: &mut u8) -> unsafe<> *mut u8 {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn mut_ptr_to_const(x: *mut u8) -> unsafe<> *const u8 {
    unsafe { wrap_binder!(x) }
    //~^ ERROR mismatched types
}

fn target() {}

fn fn_item() -> unsafe<> fn() {
    unsafe { wrap_binder!(target) }
    //~^ ERROR mismatched types
}

fn closure() -> unsafe<> fn() {
    unsafe { wrap_binder!(|| {}) }
    //~^ ERROR mismatched types
}

fn safe_to_unsafe_fn(f: fn()) -> unsafe<> unsafe fn() {
    unsafe { wrap_binder!(f) }
    //~^ ERROR mismatched types
}

fn never() -> unsafe<> u8 {
    unsafe { wrap_binder!(panic!()) }
}

fn higher_ranked_fn_ptr(f: for<'b> fn(&'b u8)) -> unsafe<'a> fn(&'a u8) {
    unsafe { wrap_binder!(f) }
}

fn unwrap_reborrow(b: unsafe<'a> &'a mut u8) -> &'static u8 {
    unsafe { unwrap_binder!(b) }
}

fn unwrap_unsize(b: unsafe<'a> &'a [u8; 2]) -> &'static [u8] {
    unsafe { unwrap_binder!(b) }
}

fn unwrap_unsize_dyn(b: unsafe<'a> &'a u8) -> &'static dyn Debug {
    unsafe { unwrap_binder!(b) }
}

fn unwrap_deref(b: unsafe<'a> &'a String) -> &'static str {
    unsafe { unwrap_binder!(b) }
}

fn unwrap_ref_to_ptr(b: unsafe<'a> &'a u8) -> *const u8 {
    unsafe { unwrap_binder!(b) }
}

fn unwrap_safe_to_unsafe_fn(b: unsafe<> fn()) -> unsafe fn() {
    unsafe { unwrap_binder!(b) }
}

fn unwrap_operand_ref(b: &unsafe<'a> &'a u8) -> u8 {
    unsafe { *unwrap_binder!(b) }
    //~^ ERROR expected unsafe binder, found `&unsafe<'a> &'a u8` as input of `unwrap_binder!()`
}

fn main() {}
