// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unsafe_ffi_drop_implementations)]
#![allow(dead_code)]

extern {
    fn f(x: *const FfiUnsafeStruct, y: *const FfiUnsafeEnum);
}

#[repr(C)]
struct FfiUnsafeStruct { //~ ERROR: unexpected size and layout
    i: i32,
}

impl Drop for FfiUnsafeStruct {
    fn drop(&mut self) {}
}

#[repr(C)]
enum FfiUnsafeEnum { //~ ERROR: unexpected size and layout
    Kaboom = 0,
    Splang = 1,
}

impl Drop for FfiUnsafeEnum {
    fn drop(&mut self) {}
}

fn main() {}
