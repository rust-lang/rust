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

// A one-member aggregate containing a float is passed in a floating point register.
#[repr(C)]
pub struct OneFloatStruct {
    pub a: f32,
}
impl Copy for OneFloatStruct {}

// ppc64: define void @test_struct_one_float(float %0)
// ppc64_vsx: define void @test_struct_one_float(float %0)
// ppc64le: define void @test_struct_one_float(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_struct_one_float(a: OneFloatStruct) {
    hint::black_box(a);
}

// A union with several members is not a one-member aggregate, so on ELFv1 it is not passed in a
// floating point register. See https://github.com/rust-lang/rust/issues/162011.
#[repr(C)]
pub union TwoFloats {
    pub a: f32,
    pub b: f32,
}

// ppc64: define void @test_union_two_floats(i32 %0)
// ppc64_vsx: define void @test_union_two_floats(i32 %0)
// ppc64le: define void @test_union_two_floats(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_union_two_floats(a: TwoFloats) {
    hint::black_box(a);
}

// A zero-sized union contributes no data, so it doesn't disqualify the aggregate.
#[repr(C)]
pub union ZstUnion {
    pub a: (),
}

#[repr(C)]
pub struct OneFloatAndZstUnion {
    pub a: f32,
    pub u: ZstUnion,
}

// ppc64: define void @test_struct_one_float_and_zst_union(float %0)
// ppc64_vsx: define void @test_struct_one_float_and_zst_union(float %0)
// ppc64le: define void @test_struct_one_float_and_zst_union(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_struct_one_float_and_zst_union(a: OneFloatAndZstUnion) {
    hint::black_box(a);
}

// On ELFv1 unions are not considered aggregates for the ABI, so they are never passed in a
// floating point register, not even when they have a single field or are nested within an
// aggregate. See https://github.com/rust-lang/rust/issues/162011.
#[repr(C)]
pub union OneFloat {
    pub a: f32,
}

// ppc64: define void @test_union_one_float(i32 %0)
// ppc64_vsx: define void @test_union_one_float(i32 %0)
// ppc64le: define void @test_union_one_float(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_union_one_float(a: OneFloat) {
    hint::black_box(a);
}

#[repr(C)]
pub struct StructOfUnion {
    pub u: OneFloat,
}

// ppc64: define void @test_struct_of_union(i32 %0)
// ppc64_vsx: define void @test_struct_of_union(i32 %0)
// ppc64le: define void @test_struct_of_union(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_struct_of_union(a: StructOfUnion) {
    hint::black_box(a);
}

#[repr(C)]
pub union UnionOfStruct {
    pub s: OneFloatStruct,
}

// ppc64: define void @test_union_of_struct(i32 %0)
// ppc64_vsx: define void @test_union_of_struct(i32 %0)
// ppc64le: define void @test_union_of_struct(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_union_of_struct(a: UnionOfStruct) {
    hint::black_box(a);
}

// A `repr(transparent)` union like `MaybeUninit` is guaranteed to be ABI-compatible with its single
// non-1-ZST field, so it is passed in a floating point register like the field would be.
//
// ppc64: define void @test_transparent_union_one_float(float %a)
// ppc64_vsx: define void @test_transparent_union_one_float(float %a)
// ppc64le: define void @test_transparent_union_one_float(float %a)
#[unsafe(no_mangle)]
pub extern "C" fn test_transparent_union_one_float(a: MaybeUninit<f32>) {
    hint::black_box(a);
}

#[repr(C)]
pub struct StructOfTransparentUnion {
    pub u: MaybeUninit<f32>,
}

// ppc64: define void @test_struct_of_transparent_union(float %0)
// ppc64_vsx: define void @test_struct_of_transparent_union(float %0)
// ppc64le: define void @test_struct_of_transparent_union(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_struct_of_transparent_union(a: StructOfTransparentUnion) {
    hint::black_box(a);
}

// A `repr(transparent)` union whose field is itself a union is still not passed in a floating
// point register on ELFv1.
//
// ppc64: define void @test_transparent_union_of_union(i32 %0)
// ppc64_vsx: define void @test_transparent_union_of_union(i32 %0)
// ppc64le: define void @test_transparent_union_of_union(float %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_transparent_union_of_union(a: MaybeUninit<OneFloat>) {
    hint::black_box(a);
}
