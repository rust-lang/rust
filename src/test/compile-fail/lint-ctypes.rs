// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(improper_ctypes)]

extern crate libc;

#[deriving(Copy)]
pub struct Bad {
    _x: u32
}

#[deriving(Copy)]
#[repr(C)]
pub struct Good {
    _x: u32
}

extern {
    pub fn bare_type1(size: int); //~ ERROR: found rust type
    pub fn bare_type2(size: uint); //~ ERROR: found rust type
    pub fn ptr_type1(size: *const int); //~ ERROR: found rust type
    pub fn ptr_type2(size: *const uint); //~ ERROR: found rust type
    pub fn non_c_repr_type(b: Bad); //~ ERROR: found type without foreign-function-safe

    pub fn good1(size: *const libc::c_int);
    pub fn good2(size: *const libc::c_uint);
    pub fn good3(g: Good);
}

pub extern fn ex_bare_type1(_: int) {} //~ ERROR: found rust type
pub extern fn ex_bare_type2(_: uint)  {} //~ ERROR: found rust type
pub extern fn ex_ptr_type1(_: *const int) {} //~ ERROR: found rust type
pub extern fn ex_ptr_type2(_: *const uint) {}  //~ ERROR: found rust type
pub extern fn ex_non_c_repr_type(_: Bad) {} //~ ERROR: found type without foreign-function-safe

pub extern fn ex_good1(_: *const libc::c_int) {}
pub extern fn ex_good2(_: *const libc::c_uint) {}
pub extern fn ex_good3(_: Good) {}

fn main() {
}
