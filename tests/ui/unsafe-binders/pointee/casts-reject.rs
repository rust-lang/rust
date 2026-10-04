//@ compile-flags: -Znext-solver

// Casts fwith unsafe binders should not change the metadata kind or the
// principal trait, or add auto traits.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::fmt::{Debug, Display};
use std::mem::ManuallyDrop as MD;

fn thin_to_wide(p: *const u8) -> *const unsafe<> MD<[u8]> {
    p as _ //~ ERROR cannot cast thin pointer
}
fn slice_to_dyn(p: *const MD<[u8]>) -> *const unsafe<'a> MD<dyn Debug + 'a> {
    p as _ //~ ERROR is invalid
}
fn change_principal(p: *const MD<dyn Debug>) -> *const unsafe<'a> MD<dyn Display + 'a> {
    p as _ //~ ERROR is invalid
}
fn add_auto_trait(p: *const MD<dyn Debug>) -> *const unsafe<'a> MD<dyn Debug + Send + 'a> {
    p as _ //~ ERROR cannot add auto trait `Send`
}

fn main() {}
