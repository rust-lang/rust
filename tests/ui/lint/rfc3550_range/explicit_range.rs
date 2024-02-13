#![crate_type="lib"]

#![deny(explicit_range)]
#![allow(dead_code)]

use std::ops::{Range, RangeInclusive, RangeFrom};

fn p(_: Range<usize>) { } // Private, no error.

pub fn f_std(_: Range<usize>) { } //~ ERROR explicit usage of range type
pub fn f_inc(_: RangeInclusive<usize>) { } //~ ERROR explicit usage of range type
pub fn f_from(_: RangeFrom<usize>) { } //~ ERROR explicit usage of range type

pub fn f_core(_: core::ops::Range<usize>) { } //~ ERROR explicit usage of range type
pub fn f_arr(_: [Range<usize>; 2]) { } //~ ERROR explicit usage of range type
pub fn f_slice(_: &[Range<usize>]) { } //~ ERROR explicit usage of range type
pub fn f_ptr(_: *const Range<usize>) { } //~ ERROR explicit usage of range type
pub fn f_ref(_: &Range<usize>) { } //~ ERROR explicit usage of range type
pub fn f_tup(_: (u8, Range<usize>)) { } //~ ERROR explicit usage of range type
pub fn f_wrapped(_: std::mem::MaybeUninit<Range<usize>>) {} //~ ERROR explicit usage of range type
pub fn f_generic<T>(_: Range<T>) {} //~ ERROR explicit usage of range type

pub trait Foo {
    fn foo(self);
    fn goo(this: Self);
    fn bar(range: Range<usize>);
}

impl Foo for Range<u8> {
    fn foo(self) {} // Detected by `trait_impl_range`
    fn goo(this: Self) {} // Detected by `trait_impl_range`
    fn bar(range: Range<usize>) {} //~ ERROR explicit usage of range type
}
