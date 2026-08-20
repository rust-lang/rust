//@ compile-flags: -Copt-level=3
//@ ignore-backends: gcc
//@ ignore-s390x
//@ ignore-wasm
//@ ignore-x86

#![crate_type = "lib"]
#![feature(f16b)]
#![allow(improper_ctypes_definitions)]

extern crate core;

use core::num::f16b;

// CHECK-LABEL: define{{.*}} bfloat @identity_f16b(bfloat
#[no_mangle]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    // CHECK: ret bfloat
    value
}

// CHECK-LABEL: define{{.*}} i16 @f16b_to_bits(bfloat
#[no_mangle]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    // CHECK: bitcast bfloat %value to i16
    value.to_bits()
}

// CHECK-LABEL: define{{.*}} bfloat @f16b_from_bits(i16
#[no_mangle]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    // CHECK: bitcast i16 %bits to bfloat
    f16b::from_bits(bits)
}

// CHECK-LABEL: define{{.*}} float @widen_f16b(bfloat
#[no_mangle]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    // CHECK: bitcast bfloat %value to i16
    // CHECK: zext i16
    // CHECK: shl nuw i32 {{.*}}, 16
    // CHECK: bitcast i32 {{.*}} to float
    f32::from(value)
}
