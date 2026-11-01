//@ add-minicore
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0
//
//@ revisions: linux win
//@[linux] compile-flags: --target aarch64-unknown-linux-gnu
//@[win] compile-flags: --target aarch64-pc-windows-msvc
//
//@ needs-llvm-components: aarch64

// Test that homogeneous aggregates are passed and returned with the correct ABI.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::simd::*;
use minicore::*;

// A homogeneous float aggregate.
#[repr(C)]
pub struct Hfa {
    pub a: f32,
    pub b: f32,
}
impl Copy for Hfa {}

// CHECK: define void @test_hfa([2 x float] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa(a: Hfa) {
    hint::black_box(a);
}

// Fields can be vectors too.
#[repr(C)]
pub struct Hfa2V2F64 {
    pub a: f64x2,
    pub b: f64x2,
}

// CHECK: define void @test_hfa_2_f64x2([2 x <2 x double>] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa_2_f64x2(a: Hfa2V2F64) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Hfa2V2U64 {
    pub a: u64x2,
    pub b: u64x2,
}

// CHECK: define void @test_hfa_2_u64x2([2 x <16 x i8>] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa_2_u64x2(a: Hfa2V2U64) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Hfa2V2F32 {
    pub a: f32x2,
    pub b: f32x2,
}

// CHECK: define void @test_hfa_2_f32x2([2 x <2 x float>] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa_2_f32x2(a: Hfa2V2F32) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Hfa4V2F64 {
    pub a: f64x2,
    pub b: f64x2,
    pub c: f64x2,
    pub d: f64x2,
}

// CHECK: define void @test_hfa_4_f64x2([4 x <2 x double>] %0)
#[unsafe(no_mangle)]
#[target_feature(enable = "neon")]
pub extern "C" fn test_hfa_4_f64x2(a: Hfa4V2F64) {
    hint::black_box(a);
}
