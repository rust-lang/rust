// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variables)]
#![deny(improper_ctypes)]
#![feature(libc)]

use types::*;

extern crate libc;

pub mod types {
    pub trait Mirror { type It; }
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
    pub trait Bar {}
}

extern {
    pub fn ptr_type1(size: *const Foo); //~ ERROR: found struct without
    pub fn ptr_type2(size: *const Foo); //~ ERROR: found struct without
    pub fn slice_type(p: &[u32]); //~ ERROR: found Rust slice type
    pub fn str_type(p: &str); //~ ERROR: found Rust type
    pub fn box_type(p: Box<u32>); //~ ERROR found Rust type
    pub fn char_type(p: char); //~ ERROR found Rust type
    pub fn trait_type(p: &Bar); //~ ERROR found Rust trait type
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

pub mod extern_fn {
    use libc;
    use types::*;

    pub extern fn ptr_type1(size: *const Foo) {} //~ ERROR: found struct without
    pub extern fn ptr_type2(size: *const Foo) {} //~ ERROR: found struct without
    pub extern fn slice_type(p: &[u32]) {} //~ ERROR: found Rust slice type
    pub extern fn str_type(p: &str) {} //~ ERROR: found Rust type
    pub extern fn box_type(p: Box<u32>) {} //~ ERROR found Rust type
    pub extern fn char_type(p: char) {} //~ ERROR found Rust type
    pub extern fn trait_type(p: &Bar) {} //~ ERROR found Rust trait type
    pub extern fn tuple_type(p: (i32, i32)) {} //~ ERROR found Rust tuple type
    pub extern fn tuple_type2(p: I32Pair) {} //~ ERROR found Rust tuple type
    pub extern fn zero_size(p: ZeroSize) {} //~ ERROR found zero-size struct
    pub extern fn fn_type(p: RustFn) {} //~ ERROR found function pointer with Rust
    pub extern fn fn_type2(p: fn()) {} //~ ERROR found function pointer with Rust
    pub extern fn fn_contained(p: RustBadRet) {} //~ ERROR: found Rust type

    pub extern fn good1(size: *const libc::c_int) {}
    pub extern fn good2(size: *const libc::c_uint) {}
    pub extern fn good3(fptr: Option<extern fn()>) {}
    pub extern fn good4(aptr: &[u8; 4 as usize]) {}
    pub extern fn good5(s: StructWithProjection) {}
    pub extern fn good6(s: StructWithProjectionAndLifetime) {}
    pub extern fn good7(fptr: extern fn() -> ()) {}
    pub extern fn good8(fptr: extern fn() -> !) {}
    pub extern fn good9() -> () {}
    pub extern fn good10() -> CVoidRet {}
    pub extern fn good11(size: isize) {}
    pub extern fn good12(size: usize) {}
}

fn main() {
}
