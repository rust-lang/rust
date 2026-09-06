//@ add-minicore
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0
//
//@ revisions: ppc64 ppc64_vsx ppc64le
//@[ppc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[ppc64_vsx] compile-flags: --target powerpc64-unknown-linux-gnu -Ctarget-feature=+vsx
//@[ppc64le] compile-flags: --target powerpc64le-unknown-linux-gnu
//
//@ needs-llvm-components: powerpc

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

// ppc64: define void @test_hfa(i64 %0)
// ppc64_vsx: define void @test_hfa(i64 %0)
// ppc64le: define void @test_hfa([2 x float] %0)
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

// ppc64: define void @test_hfa_2_f64x2([2 x i128] %0)
// ppc64_vsx: define void @test_hfa_2_f64x2([2 x i128] %0)
// ppc64le: define void @test_hfa_2_f64x2([2 x <2 x double>] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa_2_f64x2(a: Hfa2V2F64) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Hfa2V2U64 {
    pub a: u64x2,
    pub b: u64x2,
}

// ppc64: define void @test_hfa_2_u64x2([2 x i128] %0)
// ppc64_vsx: define void @test_hfa_2_u64x2([2 x i128] %0)
// ppc64le: define void @test_hfa_2_u64x2([2 x <16 x i8>] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa_2_u64x2(a: Hfa2V2U64) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Hfa2V2F32 {
    pub a: f32x2,
    pub b: f32x2,
}

// On PowerPC only 128-bit units are eligible for HVA.
//
// ppc64: define void @test_hfa_2_f32x2([2 x i64] %0)
// ppc64_vsx: define void @test_hfa_2_f32x2([2 x i64] %0)
// ppc64le: define void @test_hfa_2_f32x2([2 x i64] %0)
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

// ppc64: define void @test_hfa_4_f64x2([4 x i128] %0)
// ppc64_vsx: define void @test_hfa_4_f64x2([4 x i128] %0)
// ppc64le: define void @test_hfa_4_f64x2([4 x <2 x double>] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa_4_f64x2(a: Hfa4V2F64) {
    hint::black_box(a);
}
