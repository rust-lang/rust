//@ add-minicore
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0
//
//@ revisions: linux eabi watchos
//@[linux] compile-flags: --target armv7-unknown-linux-gnueabihf
//@[eabi] compile-flags: --target armv7r-none-eabi
//@[watchos] compile-flags: --target armv7k-apple-watchos
//
//@ needs-llvm-components: arm

// Test that homogeneous aggregates are passed and returned with the correct ABI on 32-bit arm.

#![feature(no_core, lang_items)]
#![feature(arm_target_feature)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::simd::*;
use minicore::*;

// A homogeneous float aggregate, which a hard-float ABI passes in VFP registers.
#[repr(C)]
pub struct Hfa {
    pub a: f32,
    pub b: f32,
}
impl Copy for Hfa {}

// linux:   define void @test_hfa([2 x float] %0)
// eabi:    define dso_local void @test_hfa([2 x i32] %0)
// watchos: define void @test_hfa([2 x float] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa(a: Hfa) {
    hint::black_box(a);
}

// linux:   define [2 x float] @ret_hfa([2 x float] %0)
// eabi:    define dso_local void @ret_hfa(ptr sret([8 x i8]) align 4 %_0, [2 x i32] %0)
// watchos: define [2 x float] @ret_hfa([2 x float] %0)
#[unsafe(no_mangle)]
pub extern "C" fn ret_hfa(a: Hfa) -> Hfa {
    a
}

// When we ask for `extern "aapcs"` specifically, GPRs are used, except on watchOS.
//
// linux:   define arm_aapcscc void @test_hfa_aapcs([2 x i32] %0)
// eabi:    define dso_local arm_aapcscc void @test_hfa_aapcs([2 x i32] %0)
// watchos: define arm_aapcscc void @test_hfa_aapcs([2 x float] %0)
#[unsafe(no_mangle)]
pub extern "aapcs" fn test_hfa_aapcs(a: Hfa) {
    hint::black_box(a);
}

// A homogeneous aggregate can have at most 4 fields.
#[repr(C)]
pub struct Hfa4F64 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

// linux:   define void @test_hfa_4_f64([4 x double] %0)
// eabi:    define dso_local void @test_hfa_4_f64([4 x i64] %0)
// watchos: define void @test_hfa_4_f64([4 x double] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_hfa_4_f64(a: Hfa4F64) {
    hint::black_box(a);
}

// Fields can be vectors too.
#[repr(C)]
pub struct Hfa2V2F64 {
    pub a: f64x2,
    pub b: f64x2,
}

// linux: define void @test_hfa_2_f64x2([2 x <2 x double>] %0)
// eabi: define dso_local void @test_hfa_2_f64x2([4 x i64] %0)
// watchos: define void @test_hfa_2_f64x2([2 x <2 x double>] %0)
#[unsafe(no_mangle)]
#[target_feature(enable = "neon")]
pub extern "C" fn test_hfa_2_f64x2(a: Hfa2V2F64) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Hfa2V2U64 {
    pub a: u64x2,
    pub b: u64x2,
}

// linux: define void @test_hfa_2_u64x2([2 x <16 x i8>] %0)
// eabi: define dso_local void @test_hfa_2_u64x2([4 x i64] %0)
// watchos: define void @test_hfa_2_u64x2([2 x <16 x i8>] %0)
#[unsafe(no_mangle)]
#[target_feature(enable = "neon")]
pub extern "C" fn test_hfa_2_u64x2(a: Hfa2V2U64) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Hfa2V2F32 {
    pub a: f32x2,
    pub b: f32x2,
}

// linux: define void @test_hfa_2_f32x2([2 x <2 x float>] %0)
// eabi: define dso_local void @test_hfa_2_f32x2([2 x i64] %0)
// watchos: define void @test_hfa_2_f32x2([2 x <2 x float>] %0)
#[unsafe(no_mangle)]
#[target_feature(enable = "neon")]
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

// linux: define void @test_hfa_4_f64x2([4 x <2 x double>] %0)
// eabi: define dso_local void @test_hfa_4_f64x2([8 x i64] %0)
// watchos: define void @test_hfa_4_f64x2([4 x <2 x double>] %0)
#[unsafe(no_mangle)]
#[target_feature(enable = "neon")]
pub extern "C" fn test_hfa_4_f64x2(a: Hfa4V2F64) {
    hint::black_box(a);
}

// A homogeneous aggregate can have at most 4 fields, so this does not qualify.
#[repr(C)]
pub struct Floats5 {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
}

// linux:   define void @test_floats_5([5 x i32] %0)
// eabi:    define dso_local void @test_floats_5([5 x i32] %0)
// watchos: define void @test_floats_5(ptr align 4 %a)
#[unsafe(no_mangle)]
pub extern "C" fn test_floats_5(a: Floats5) {
    hint::black_box(a);
}

// Just a big struct, note how watchOS passes it indirectly.
#[repr(C)]
pub struct Ints6 {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub e: u32,
    pub f: u32,
}
impl Copy for Ints6 {}

// linux:   define void @test_ints_6([6 x i32] %0)
// eabi:    define dso_local void @test_ints_6([6 x i32] %0)
// watchos: define void @test_ints_6(ptr align 4 %a)
#[unsafe(no_mangle)]
pub extern "C" fn test_ints_6(a: Ints6) {
    hint::black_box(a);
}

#[repr(C)]
pub struct Natural {
    pub a: u32,
    pub b: u32,
}

// Returns are indirect once they exceed 32 bits, except on watchOS.
//
// linux:   define void @ret_natural(ptr sret([8 x i8]) align 4 %_0, [2 x i32] %0)
// eabi:    define dso_local void @ret_natural(ptr sret([8 x i8]) align 4 %_0, [2 x i32] %0)
// watchos: define [2 x i32] @ret_natural([2 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn ret_natural(a: Natural) -> Natural {
    a
}

#[repr(C)]
#[repr(align(16))]
pub struct Align16 {
    pub a: u32,
    pub b: u32,
}

// linux:   define void @ret_align16(ptr sret([16 x i8]) align 16 %_0, [4 x i32] %0)
// eabi:    define dso_local void @ret_align16(ptr sret([16 x i8]) align 16 %_0, [4 x i32] %0)
// watchos: define [4 x i32] @ret_align16([2 x i64] %0)
#[unsafe(no_mangle)]
pub extern "C" fn ret_align16(a: Align16) -> Align16 {
    a
}

// Larger than 128 bits, so the return is indirect everywhere.
//
// linux:   define void @ret_ints_6(ptr sret([24 x i8]) align 4 %_0, [6 x i32] %0)
// eabi:    define dso_local void @ret_ints_6(ptr sret([24 x i8]) align 4 %_0, [6 x i32] %0)
// watchos: define void @ret_ints_6(ptr sret([24 x i8]) align 4 %_0, ptr align 4 %a)
#[unsafe(no_mangle)]
pub extern "C" fn ret_ints_6(a: Ints6) -> Ints6 {
    a
}

extern "C" {
    // linux:   declare void @test_hfa_variadic_c([2 x i32], ...)
    // eabi:    declare dso_local void @test_hfa_variadic_c([2 x i32], ...)
    // watchos: declare void @test_hfa_variadic_c([2 x float], ...)
    fn test_hfa_variadic_c(_: Hfa, ...);

    // linux:   declare void @test_ints_6_variadic_c([6 x i32], ...)
    // eabi:    declare dso_local void @test_ints_6_variadic_c([6 x i32], ...)
    // watchos: declare void @test_ints_6_variadic_c(ptr align 4, ...)
    fn test_ints_6_variadic_c(_: Ints6, ...);
}

extern "aapcs" {
    // linux:   declare arm_aapcscc void @test_hfa_variadic_aapcs([2 x i32], ...)
    // eabi:    declare dso_local arm_aapcscc void @test_hfa_variadic_aapcs([2 x i32], ...)
    // watchos: declare arm_aapcscc void @test_hfa_variadic_aapcs([2 x float], ...)
    fn test_hfa_variadic_aapcs(_: Hfa, ...);

    // linux:   declare arm_aapcscc void @test_ints_6_variadic_aapcs([6 x i32], ...)
    // eabi:    declare dso_local arm_aapcscc void @test_ints_6_variadic_aapcs([6 x i32], ...)
    // watchos: declare arm_aapcscc void @test_ints_6_variadic_aapcs(ptr align 4, ...)
    fn test_ints_6_variadic_aapcs(_: Ints6, ...);
}

pub unsafe fn call_variadics(a: Hfa, b: Ints6) {
    test_hfa_variadic_c(a);
    test_ints_6_variadic_c(b);

    test_hfa_variadic_aapcs(a);
    test_ints_6_variadic_aapcs(b);
}
