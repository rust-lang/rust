//@ check-pass

#![feature(ptr_metadata)]

use std::ptr::Thin;

fn main() {}

fn foo<T: ?Sized + Thin>(t: *const T) -> *const () {
    unsafe { std::mem::transmute(t) }
}
