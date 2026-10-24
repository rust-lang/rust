//@ compile-flags: -Copt-level=3
//@ revisions: GENERIC LOONGARCH32 LOONGARCH64 RISCV32 RISCV64
//@ [GENERIC] ignore-loongarch32
//@ [GENERIC] ignore-loongarch64
//@ [GENERIC] ignore-riscv32
//@ [GENERIC] ignore-riscv64
//@ [LOONGARCH32] only-loongarch32
//@ [LOONGARCH64] only-loongarch64
//@ [RISCV32] only-riscv32
//@ [RISCV64] only-riscv64

#![crate_type = "lib"]
#![feature(f16b)]
#![allow(improper_ctypes_definitions)]

extern crate core;

use core::num::f16b;

// GENERIC-LABEL: define{{.*}} bfloat @identity_f16b(bfloat
// LOONGARCH32-LABEL: define{{.*}} half @identity_f16b(half
// LOONGARCH64-LABEL: define{{.*}} half @identity_f16b(half
// RISCV32-LABEL: define{{.*}} half @identity_f16b(half
// RISCV64-LABEL: define{{.*}} half @identity_f16b(half
#[no_mangle]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    // GENERIC: ret bfloat
    // LOONGARCH32: ret half
    // LOONGARCH64: ret half
    // RISCV32: ret half
    // RISCV64: ret half
    value
}

// GENERIC-LABEL: define{{.*}} i16 @f16b_to_bits(bfloat
// LOONGARCH32-LABEL: define{{.*}} i16 @f16b_to_bits(half
// LOONGARCH64-LABEL: define{{.*}} i16 @f16b_to_bits(half
// RISCV32-LABEL: define{{.*}} i16 @f16b_to_bits(half
// RISCV64-LABEL: define{{.*}} i16 @f16b_to_bits(half
#[no_mangle]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    // GENERIC: bitcast bfloat %value to i16
    // LOONGARCH32: bitcast half %0 to i16
    // LOONGARCH64: bitcast half %0 to i16
    // RISCV32: bitcast half %0 to i16
    // RISCV64: bitcast half %0 to i16
    value.to_bits()
}

// GENERIC-LABEL: define{{.*}} bfloat @f16b_from_bits(i16
// LOONGARCH32-LABEL: define{{.*}} half @f16b_from_bits(i16
// LOONGARCH64-LABEL: define{{.*}} half @f16b_from_bits(i16
// RISCV32-LABEL: define{{.*}} half @f16b_from_bits(i16
// RISCV64-LABEL: define{{.*}} half @f16b_from_bits(i16
#[no_mangle]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    // GENERIC: bitcast i16 %bits to bfloat
    // LOONGARCH32: bitcast i16 %bits to half
    // LOONGARCH64: bitcast i16 %bits to half
    // RISCV32: bitcast i16 %bits to half
    // RISCV64: bitcast i16 %bits to half
    f16b::from_bits(bits)
}

// GENERIC-LABEL: define{{.*}} float @widen_f16b(bfloat
// LOONGARCH32-LABEL: define{{.*}} float @widen_f16b(half
// LOONGARCH64-LABEL: define{{.*}} float @widen_f16b(half
// RISCV32-LABEL: define{{.*}} float @widen_f16b(half
// RISCV64-LABEL: define{{.*}} float @widen_f16b(half
#[no_mangle]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    // GENERIC: bitcast bfloat %value to i16
    // LOONGARCH32: bitcast half %0 to i16
    // LOONGARCH64: bitcast half %0 to i16
    // RISCV32: bitcast half %0 to i16
    // RISCV64: bitcast half %0 to i16
    // CHECK: zext i16
    // CHECK: shl nuw i32 {{.*}}, 16
    // CHECK: bitcast i32 {{.*}} to float
    f32::from(value)
}
