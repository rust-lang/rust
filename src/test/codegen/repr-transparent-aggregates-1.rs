// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

// ignore-arm
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// See repr-transparent.rs

#![crate_type="lib"]


#[repr(C)]
pub struct Big([u32; 16]);

#[repr(transparent)]
pub struct BigW(Big);

// CHECK: define void @test_Big(%Big* [[BIG_RET_ATTRS:.*]], %Big* [[BIG_ARG_ATTRS:.*]])
#[no_mangle]
pub extern fn test_Big(_: Big) -> Big { loop {} }

// CHECK: define void @test_BigW(%BigW* [[BIG_RET_ATTRS]], %BigW* [[BIG_ARG_ATTRS]])
#[no_mangle]
pub extern fn test_BigW(_: BigW) -> BigW { loop {} }


#[repr(C)]
pub union BigU {
    foo: [u32; 16],
}

#[repr(transparent)]
pub struct BigUw(BigU);

// CHECK: define void @test_BigU(%BigU* [[BIGU_RET_ATTRS:.*]], %BigU* [[BIGU_ARG_ATTRS:.*]])
#[no_mangle]
pub extern fn test_BigU(_: BigU) -> BigU { loop {} }

// CHECK: define void @test_BigUw(%BigUw* [[BIGU_RET_ATTRS]], %BigUw* [[BIGU_ARG_ATTRS]])
#[no_mangle]
pub extern fn test_BigUw(_: BigUw) -> BigUw { loop {} }
