// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#[repr(packed)]
pub struct Packed {
    dealign: u8,
    data: u32
}

// CHECK-LABEL: @write_pkd
#[no_mangle]
pub fn write_pkd(pkd: &mut Packed) -> u32 {
// CHECK: %{{.*}} = load i32, i32* %{{.*}}, align 1
// CHECK: store i32 42, i32* %{{.*}}, align 1
    let result = pkd.data;
    pkd.data = 42;
    result
}

pub struct Array([i32; 8]);
#[repr(packed)]
pub struct BigPacked {
    dealign: u8,
    data: Array
}

// CHECK-LABEL: @call_pkd
#[no_mangle]
pub fn call_pkd(f: fn() -> Array) -> BigPacked {
// CHECK: [[ALLOCA:%[_a-z0-9]+]] = alloca %Array
// CHECK: call void %{{.*}}(%Array* noalias nocapture sret dereferenceable(32) [[ALLOCA]])
// CHECK: call void @llvm.memcpy.{{.*}}(i8* %{{.*}}, i8* %{{.*}}, i{{[0-9]+}} 32, i32 1, i1 false)
    // check that calls whose destination is a field of a packed struct
    // go through an alloca rather than calling the function with an
    // unaligned destination.
    BigPacked { dealign: 0, data: f() }
}

#[repr(packed)]
#[derive(Copy, Clone)]
pub struct PackedPair(u8, u32);

// CHECK-LABEL: @pkd_pair
#[no_mangle]
pub fn pkd_pair(pair1: &mut PackedPair, pair2: &mut PackedPair) {
// CHECK: call void @llvm.memcpy.{{.*}}(i8* %{{.*}}, i8* %{{.*}}, i{{[0-9]+}} 5, i32 1, i1 false)
    *pair2 = *pair1;
}

#[repr(packed)]
#[derive(Copy, Clone)]
pub struct PackedNestedPair((u32, u32));

// CHECK-LABEL: @pkd_nested_pair
#[no_mangle]
pub fn pkd_nested_pair(pair1: &mut PackedNestedPair, pair2: &mut PackedNestedPair) {
// CHECK: call void @llvm.memcpy.{{.*}}(i8* %{{.*}}, i8* %{{.*}}, i{{[0-9]+}} 8, i32 1, i1 false)
    *pair2 = *pair1;
}
