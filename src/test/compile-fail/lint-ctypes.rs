// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(improper_ctypes)]
#![feature(libc)]

extern crate libc;

trait Mirror { type It; }
impl<T> Mirror for T { type It = Self; }
#[repr(C)]
pub struct StructWithProjection(*mut <StructWithProjection as Mirror>::It);
#[repr(C)]
pub struct StructWithProjectionAndLifetime<'a>(
    &'a mut <StructWithProjectionAndLifetime<'a> as Mirror>::It
);
pub type I32Pair = (i32, i32);
#[repr(C)]
pub struct ZeroSize;
pub type RustFn = fn();
pub type RustBadRet = extern fn() -> Box<u32>;
pub type CVoidRet = ();
pub struct Foo;

extern {
    pub fn ptr_type1(size: *const Foo); //~ ERROR: found struct without
    pub fn ptr_type2(size: *const Foo); //~ ERROR: found struct without
    pub fn slice_type(p: &[u32]); //~ ERROR: found Rust slice type
    pub fn str_type(p: &str); //~ ERROR: found Rust type
    pub fn box_type(p: Box<u32>); //~ ERROR found Rust type
    pub fn char_type(p: char); //~ ERROR found Rust type
    pub fn trait_type(p: &Clone); //~ ERROR found Rust trait type
    pub fn tuple_type(p: (i32, i32)); //~ ERROR found Rust tuple type
    pub fn tuple_type2(p: I32Pair); //~ ERROR found Rust tuple type
    pub fn zero_size(p: ZeroSize); //~ ERROR found zero-size struct
    pub fn fn_type(p: RustFn); //~ ERROR found function pointer with Rust
    pub fn fn_type2(p: fn()); //~ ERROR found function pointer with Rust
    pub fn fn_contained(p: RustBadRet); //~ ERROR: found Rust type

    pub fn good1(size: *const libc::c_int);
    pub fn good2(size: *const libc::c_uint);
    pub fn good3(fptr: Option<extern fn()>);
    pub fn good4(aptr: &[u8; 4 as usize]);
    pub fn good5(s: StructWithProjection);
    pub fn good6(s: StructWithProjectionAndLifetime);
    pub fn good7(fptr: extern fn() -> ());
    pub fn good8(fptr: extern fn() -> !);
    pub fn good9() -> ();
    pub fn good10() -> CVoidRet;
    pub fn good11(size: isize);
    pub fn good12(size: usize);
}

fn main() {
}
