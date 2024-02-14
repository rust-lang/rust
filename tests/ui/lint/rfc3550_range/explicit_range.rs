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

const PRIV_CONST: Range<usize> = 1..3;
static PRIV_STATIC: Range<usize> = 1..3;
type PrivAlias = Range<u8>;

pub const PUB_CONST: Range<usize> = 1..3; //~ ERROR explicit usage of range type
pub static PUB_STATIC: Range<usize> = 1..3; //~ ERROR explicit usage of range type
pub type PubAlias = Range<u8>; //~ ERROR explicit usage of range type

enum PrivEnumPubField {
    A(Range<u32>),
    B(u8),
}
struct PrivStructPubField {
    pub pub_field: Range<u32>,
}

pub struct PubStructPrivField {
    priv_field: Range<u32>,
}

pub enum PubEnum {
    A(Range<u32>), //~ ERROR explicit usage of range type
    B(u8),
}
pub struct PubStruct {
    pub pub_field: Range<u32>, //~ ERROR explicit usage of range type
    pub wrapped: std::mem::MaybeUninit<Range<usize>>, //~ ERROR explicit usage of range type
}
