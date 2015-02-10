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
    fn f(x: *const FfiSafeStruct, y: *const FfiSafeEnum);
}

#[repr(C)]
#[unsafe_no_drop_flag]
struct FfiSafeStruct {
    i: i32,
}

impl Drop for FfiSafeStruct {
    fn drop(&mut self) {}
}

#[repr(C)]
#[unsafe_no_drop_flag]
enum FfiSafeEnum {
    Kaboom = 0,
    Splang = 1,
}

impl Drop for FfiSafeEnum {
    fn drop(&mut self) {}
}

// These two should not be affected as they have no Drop impl.
#[repr(C)]
struct FfiSafeStructNoDrop {
    i: i32,
}

#[repr(C)]
enum FfiSafeEnumNoDrop {
    Peace = 0,
    WhaleSong = 1,
}

fn main() {}
