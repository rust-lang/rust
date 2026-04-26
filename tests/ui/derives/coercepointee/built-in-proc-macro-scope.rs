//@ check-pass
//@ proc-macro: another-proc-macro.rs
//@ compile-flags: -Zunpretty=expanded
//@ edition:2015

#![feature(derive_coerce_pointee)]

#[macro_use]
extern crate another_proc_macro;

use another_proc_macro::{AnotherMacro, pointee};

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct Ptr<'a, #[pointee] T: ?Sized> {
    data: &'a mut T,
}

#[pointee]
fn f() {}

#[derive(AnotherMacro)]
#[pointee]
struct MyStruct;

fn main() {}
